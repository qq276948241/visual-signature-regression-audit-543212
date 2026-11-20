"""
Perceptual image hashing library

Example:

>>> from PIL import Image
>>> import imagehash
>>> hash = imagehash.average_hash(Image.open('test.png'))
>>> print(hash)
d879f8f89b1bbf
>>> otherhash = imagehash.average_hash(Image.open('other.bmp'))
>>> print(otherhash)
ffff3720200ffff
>>> print(hash == otherhash)
False
>>> print(hash - otherhash)
36
>>> for r in range(1, 30, 5):
...	rothash = imagehash.average_hash(Image.open('test.png').rotate(r))
...	print('Rotation by %d: %d Hamming difference' % (r, hash - rothash))
...
Rotation by 1: 2 Hamming difference
Rotation by 6: 11 Hamming difference
Rotation by 11: 13 Hamming difference
Rotation by 16: 17 Hamming difference
Rotation by 21: 19 Hamming difference
Rotation by 26: 21 Hamming difference
>>>
"""

from __future__ import absolute_import, division, print_function

import io
import operator
import sys
import threading

import numpy
from PIL import Image, ImageFilter, ImageOps

try:
	ANTIALIAS = Image.Resampling.LANCZOS
except AttributeError:
	# deprecated in pillow 10
	# https://pillow.readthedocs.io/en/stable/deprecations.html
	ANTIALIAS = Image.ANTIALIAS


"""
You may copy this file, if you keep the copyright information below:


Copyright (c) 2013-2022, Johannes Buchner
https://github.com/JohannesBuchner/imagehash

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

__version__ = '4.3.2'

#: Opaque colour used to flatten transparent pixels before hashing.
#:
#: Transparent pixels still carry RGB values in RGBA, PA, LA and palette
#: images.  Letting Pillow drop the alpha channel would silently expose
#: those hidden values, so ``to_visual_image`` composites every image over
#: this fixed background instead.  White keeps fully opaque images
#: bit-for-bit identical to their pre-contract hashes.
DEFAULT_BACKGROUND = (255, 255, 255)

#: Serialises frame extraction for multi-frame images.  ``Image.seek()``
#: mutates the caller's image object, so two threads reading frames from the
#: same GIF/APNG must not interleave.  This lock protects no pixel data of
#: its own.
_FRAME_LOCK = threading.RLock()

_ALPHA_MODES = frozenset(('RGBA', 'LA', 'PA', 'RGBa', 'La'))


class ImageHashError(ValueError):
	"""Base class for invalid images or hashing parameters."""


class ImageHashSizeError(ImageHashError, AssertionError):
	"""Raised for invalid ``hash_size``/``image_scale``/``binbits`` values.

	Also subclasses :class:`AssertionError` so that callers written against
	the historical ``assert`` based validation keep working.
	"""


def _check_image(image):
	"""Validate the public ``image`` argument and return it unchanged."""
	if isinstance(image, Image.Image):
		return image
	# Duck-typed ``Image.Image`` look-alikes (e.g. mocks) are accepted as long
	# as they expose the size/mode attributes the hash functions require.
	if hasattr(image, 'mode') and hasattr(image, 'size'):
		return image
	raise TypeError('image must be a PIL.Image.Image, not {!r}'.format(type(image).__name__))


def _check_int(name, value, minimum=None):
	"""Return ``value`` as a plain int, rejecting bools and non-integers."""
	if isinstance(value, bool):
		raise ImageHashSizeError('{} must be an integer, not bool'.format(name))
	try:
		number = operator.index(value)
	except TypeError:
		raise ImageHashSizeError('{} must be an integer, got {!r}'.format(name, value))
	if minimum is not None and number < minimum:
		raise ImageHashSizeError('{} must be >= {}, got {}'.format(name, minimum, number))
	return number


def _check_hash_size(hash_size, minimum=2):
	hash_size = _check_int('hash_size', hash_size, minimum)
	return hash_size


def _frame_count(image):
	"""Number of frames in ``image`` (1 for still images)."""
	try:
		count = getattr(image, 'n_frames', 1)
	except Exception:
		return 1
	if count is None:
		return 1
	return int(count)


def _independent_handle(image):
	"""Return a second PIL handle for the same underlying data, if possible.

	Pillow's APNG reader cannot serve ``seek()`` and ``copy()`` interleaved on
	the same object without eventually raising "APNG contains frame sequence
	errors", and its GIF reader closes a shared file object when a reopened
	handle is closed.  Reading the encoded bytes into a private buffer gives a
	handle that can be seeked freely without touching the caller's object.
	"""
	fp = getattr(image, 'fp', None)
	if fp is None or not hasattr(fp, 'seek') or not hasattr(fp, 'tell'):
		return None
	try:
		position = fp.tell()
		fp.seek(0)
		data = fp.read()
		fp.seek(position)
		if not data:
			return None
		clone = Image.open(io.BytesIO(data))
		clone.load()
		return clone
	except Exception:
		try:
			fp.seek(position)
		except Exception:
			pass
		return None


def _copy_frame_in_place(image, frame):
	"""Seek ``image`` to ``frame``, copy it, and restore the original frame."""
	try:
		current = image.tell()
	except Exception:
		current = 0
	try:
		image.seek(frame)
		return image.copy()
	finally:
		try:
			image.seek(current)
		except Exception:
			pass


def select_frame(image, frame=0):
	"""Return a detached *first* (or ``frame``-th) frame of ``image``.

	Multi-frame GIF/APNG inputs are handled explicitly instead of leaving the
	result to wherever the caller's iterator happens to point.  By default the
	first frame is hashed.  The return value is a standalone image: seeking
	never mutates the caller's object and the copied frame keeps its palette,
	``info`` (including GIF ``transparency``) and EXIF metadata.

	:param image: the image to read
	:param frame: 0-based frame index; negative indices count from the end
	"""
	image = _check_image(image)
	count = _frame_count(image)
	if not isinstance(frame, int) or isinstance(frame, bool):
		raise TypeError('frame must be an int')
	if frame < 0:
		frame += count
	if frame < 0 or frame >= count:
		raise ImageHashError(
			'frame {} out of range for an image with {} frame(s)'.format(frame, count)
		)
	if count == 1:
		return image.copy()
	# ``seek`` mutates the shared image, so serialise frame access.
	with _FRAME_LOCK:
		clone = _independent_handle(image)
		if clone is not None and _frame_count(clone) == count:
			try:
				clone.seek(frame)
				return clone.copy()
			finally:
				clone.close()
		return _copy_frame_in_place(image, frame)


_PRE_MULTIPLIED_MODES = frozenset(('RGBa', 'La'))


def _unpremultiply(image):
	"""Convert a premultiplied-alpha image (``RGBa``/``La``) to straight RGBA.

	Pillow cannot convert ``La`` at all and silently mis-unpremultiplies
	``RGBa``, so the stored samples are expanded here:

	``premultiplied = round(straight * alpha / 255)``  =>  ``straight = premultiplied * 255 / alpha``

	with ``straight`` clamped to 255.
	"""
	raw = numpy.frombuffer(image.tobytes(), dtype=numpy.uint8)
	raw = raw.reshape((image.size[1], image.size[0], len(image.getbands())))
	alpha = raw[..., -1].astype(numpy.int32)
	channels = raw[..., :-1].astype(numpy.int32)
	with numpy.errstate(divide='ignore', invalid='ignore'):
		straight = numpy.where(
			alpha[..., None] == 0,
			0,
			numpy.minimum(255, (channels * 255 + alpha[..., None] // 2) // numpy.maximum(alpha[..., None], 1)),
		)
	if channels.shape[2] == 1:
		straight = numpy.repeat(straight, 3, axis=2)
	else:
		straight = straight[:, :, :3]
	rgba = numpy.dstack([straight, alpha[..., None]]).astype(numpy.uint8)
	# A (h, w, 4) uint8 array infers RGBA without the deprecated mode kwarg.
	return Image.fromarray(rgba)


def _flatten_alpha(image, background):
	"""Composite ``image`` over an opaque ``background`` and return RGB.

	Transparent pixels in RGBA/LA/PA images still store RGB values that Pillow
	drops when converting straight to ``L``/``RGB``.  Palette, ``L`` and
	``RGB`` images may also declare colour-key transparency through the
	``transparency`` entry of ``Image.info``.  Compositing makes the visible
	pixel a deterministic function of the declared background rather than of
	whatever was hidden behind alpha.
	"""
	has_alpha = image.mode in _ALPHA_MODES or 'transparency' in image.info
	if not has_alpha:
		return image.convert('RGB')
	if image.mode in _PRE_MULTIPLIED_MODES:
		foreground = _unpremultiply(image)
	else:
		foreground = image.convert('RGBA')
	if background == (255, 255, 255):
		# Identical to the general path, but skips a full-size composite for
		# the default background.
		canvas = Image.new('RGBA', foreground.size, (255, 255, 255, 255))
	else:
		canvas = Image.new('RGBA', foreground.size, background + (255,))
	return Image.alpha_composite(canvas, foreground).convert('RGB')


def _background_channel(value):
	"""Validate a single 0..255 background channel."""
	try:
		channel = operator.index(value)
	except TypeError:
		raise ValueError('background channels must be integers in 0..255')
	if not 0 <= channel <= 255:
		raise ValueError('background channels must be integers in 0..255')
	return channel


def _check_background(background):
	"""Validate a background colour and return an ``(r, g, b)`` tuple."""
	if isinstance(background, str):
		try:
			return Image.new('RGB', (1, 1), background).getpixel((0, 0))
		except (ValueError, TypeError):
			raise ValueError('unknown background colour {!r}'.format(background))
	try:
		values = tuple(background)
	except TypeError:
		raise TypeError('background must be a colour name or an (r, g, b) tuple')
	if len(values) != 3:
		raise ValueError('background must have exactly 3 channels')
	return tuple(_background_channel(value) for value in values)


def to_visual_image(image, frame=0, background=DEFAULT_BACKGROUND):
	"""Normalise an arbitrary PIL image to the agreed visual RGB input.

	This is the preprocessing contract shared by every public hash function:

	1. Multi-frame GIF/APNG images are pinned to one frame (first by default);
		the caller's image is not seeked or modified.
	2. EXIF orientation is applied (``ImageOps.exif_transpose``), so a rotated
		JPEG hashes like the visual image it describes.
	3. Images with an alpha channel are composited over ``background``
		(white by default).  Palette, ``L`` and ``RGB`` images that declare
		colour-key transparency in ``Image.info`` are treated the same way.
	4. Every other mode (grayscale, palette, CMYK, ``I``, ...) is converted to
		RGB directly.

	:param image: PIL image
	:param frame: 0-based frame index for multi-frame inputs
	:param background: colour name or ``(r, g, b)`` tuple behind alpha
	:raises TypeError: if ``image`` is not a PIL image
	:raises ImageHashError: for empty images or invalid frame indices
	"""
	image = _check_image(image)
	if image.size[0] == 0 or image.size[1] == 0:
		raise ImageHashError(
			'cannot hash an empty image with size {!r}'.format(tuple(image.size))
		)
	image = select_frame(image, frame)
	image = ImageOps.exif_transpose(image)
	return _flatten_alpha(image, _check_background(background))


def to_gray_image(image, frame=0, background=DEFAULT_BACKGROUND):
	"""``to_visual_image`` followed by Pillow's ``convert('L')``.

	Keeping the RGB step explicit matters: converting RGBA straight to ``L``
	ignores alpha and exposes hidden RGB values, whereas going through the
	composited RGB yields the agreed visible luminance.
	"""
	return to_visual_image(image, frame=frame, background=background).convert('L')


def _binary_array_to_hex(arr):
	"""
	internal function to make a hex string out of a binary array.
	"""
	bit_string = ''.join(str(b) for b in 1 * arr.flatten())
	width = int(numpy.ceil(len(bit_string) / 4))
	return '{:0>{width}x}'.format(int(bit_string, 2), width=width)


class ImageHash:
	"""
	Hash encapsulation. Can be used for dictionary keys and comparisons.
	"""

	def __init__(self, binary_array):
		# type: (NDArray) -> None
		self.hash = binary_array  # type: NDArray

	def __str__(self):
		return _binary_array_to_hex(self.hash.flatten())

	def __repr__(self):
		return repr(self.hash)

	def __sub__(self, other):
		# type: (ImageHash) -> int
		if other is None:
			raise TypeError('Other hash must not be None.')
		if not isinstance(other, ImageHash):
			raise TypeError('Other hash must be an ImageHash, not {!r}'.format(type(other).__name__))
		if self.hash.size != other.hash.size:
			raise TypeError('ImageHashes must be of the same shape.', self.hash.shape, other.hash.shape)
		return numpy.count_nonzero(self.hash.flatten() != other.hash.flatten())

	def __eq__(self, other):
		# type: (object) -> bool
		# Anything that is not an ImageHash (including None) is unequal
		# rather than an error or a reflected array comparison.
		if not isinstance(other, ImageHash):
			return False
		return numpy.array_equal(self.hash.flatten(), other.hash.flatten())  # type: ignore

	def __ne__(self, other):
		# type: (object) -> bool
		return not self.__eq__(other)

	def __lt__(self, other):
		# type: (object) -> bool
		# Plain ordering so that hashes parsed from strings still sort.
		if not isinstance(other, ImageHash):
			return NotImplemented
		return self._sort_key() < other._sort_key()

	def __le__(self, other):
		# type: (object) -> bool
		if not isinstance(other, ImageHash):
			return NotImplemented
		return self._sort_key() <= other._sort_key()

	def __gt__(self, other):
		# type: (object) -> bool
		if not isinstance(other, ImageHash):
			return NotImplemented
		return self._sort_key() > other._sort_key()

	def __ge__(self, other):
		# type: (object) -> bool
		if not isinstance(other, ImageHash):
			return NotImplemented
		return self._sort_key() >= other._sort_key()

	def _sort_key(self):
		return (self.hash.size, self.hash.shape, tuple(int(b) for b in self.hash.flatten()))

	def __hash__(self):
		# this returns a 8 bit integer, intentionally shortening the information
		return sum([2**(i % 8) for i, v in enumerate(self.hash.flatten()) if v])

	def __len__(self):
		# Returns the bit length of the hash
		return self.hash.size


# dynamic code for typing
try:
	# specify allowed values if possible (py3.8+)
	from typing import Literal
	WhashMode = Literal['haar', 'db4']  # type: ignore
except ImportError:
	WhashMode = str  # type: ignore

try:
	# enable numpy array typing (py3.7+)
	import numpy.typing
	NDArray = numpy.typing.NDArray[numpy.bool_]
except (AttributeError, ImportError):
	NDArray = list  # type: ignore

# type of Callable
if sys.version_info >= (3, 3):
	if sys.version_info >= (3, 9, 0) and sys.version_info <= (3, 9, 1):
		# https://stackoverflow.com/questions/65858528/is-collections-abc-callable-bugged-in-python-3-9-1
		from typing import Callable
	else:
		from collections.abc import Callable
	try:
		MeanFunc = Callable[[NDArray], float]
		HashFunc = Callable[[Image.Image], ImageHash]
	except TypeError:
		MeanFunc = Callable  # type: ignore
		HashFunc = Callable  # type: ignore
# end of dynamic code for typing


def hex_to_hash(hexstr):
	# type: (str) -> ImageHash
	"""
	Convert a stored hash (hex, as retrieved from str(Imagehash))
	back to a Imagehash object.

	The bit width is ``int(sqrt(len(hex) * 4)) ** 2``; this keeps the result
	square for every size the library itself can emit, including odd
	``hash_size`` values that need a final hex digit with spare bits.

	Notes:
	1. This algorithm assumes all hashes are either
			bidimensional arrays with dimensions hash_size * hash_size,
			or onedimensional arrays with dimensions binbits * 14.
	2. This algorithm does not work for hash_size < 2.

	:raises ValueError: if ``hexstr`` is not valid hexadecimal, or if its
		value does not fit the square bit width implied by its length.
	"""
	try:
		value = int(hexstr, 16)
	except (TypeError, ValueError):
		raise ValueError('invalid hexadecimal hash {!r}'.format(hexstr))
	if value < 0:
		raise ValueError('invalid hexadecimal hash {!r}'.format(hexstr))
	hash_size = int(numpy.sqrt(len(hexstr) * 4))
	width = hash_size * hash_size
	if value.bit_length() > width:
		raise ValueError(
			'hexadecimal hash {!r} needs {} bits but implies a {}x{} hash'.format(
				hexstr, value.bit_length(), hash_size, hash_size)
		)
	binary_array = '{:0>{width}b}'.format(value, width=width)
	bit_rows = [binary_array[i:i + hash_size] for i in range(0, len(binary_array), hash_size)]
	hash_array = numpy.array([[bool(int(d)) for d in row] for row in bit_rows])
	return ImageHash(hash_array)


def hex_to_flathash(hexstr, hashsize):
	# type: (str, int) -> ImageHash
	"""
	Parse a hex string produced by ``str(ImageHash)`` for a flat hash.

	``hashsize`` is the number of bins (14 for colorhash).  The returned array
	has shape ``(hashsize, binbits)`` — the layout ``colorhash`` produces — so
	a value serialised as ``str(colorhash(...))`` round-trips structurally and
	not merely bit-for-bit.

	:raises ValueError: for non-hexadecimal input, a non-positive ``hashsize``,
		or a value too wide for its implied ``binbits``.
	"""
	if not isinstance(hashsize, int) or isinstance(hashsize, bool) or hashsize < 1:
		raise ValueError('hashsize must be a positive integer')
	try:
		value = int(hexstr, 16)
	except (TypeError, ValueError):
		raise ValueError('invalid hexadecimal hash {!r}'.format(hexstr))
	if value < 0:
		raise ValueError('invalid hexadecimal hash {!r}'.format(hexstr))
	hash_size = int(len(hexstr) * 4 / hashsize)
	if hash_size < 1:
		raise ValueError(
			'hexadecimal hash {!r} is too short for {} bins'.format(hexstr, hashsize)
		)
	width = hash_size * hashsize
	if value.bit_length() > width:
		raise ValueError(
			'hexadecimal hash {!r} needs {} bits but implies {} bins of {} bits'.format(
				hexstr, value.bit_length(), hashsize, hash_size)
		)
	binary_array = '{:0>{width}b}'.format(value, width=width)
	hash_array = numpy.array([bool(int(d)) for d in binary_array]).reshape((hashsize, hash_size))
	return ImageHash(hash_array)


def hex_to_multihash(hexstr):
	# type: (str) -> ImageMultiHash
	"""
	Convert a stored multihash (hex, as retrieved from str(ImageMultiHash))
	back to an ImageMultiHash object.

	This function is based on hex_to_hash so the same caveats apply. Namely:

	1. This algorithm assumes all hashes are either
			bidimensional arrays with dimensions hash_size * hash_size,
			or onedimensional arrays with dimensions binbits * 14.
	2. This algorithm does not work for hash_size < 2.
	"""
	split = hexstr.split(',')
	hashes = [hex_to_hash(x) for x in split]
	return ImageMultiHash(hashes)


def old_hex_to_hash(hexstr, hash_size=8):
	# type: (str, int) -> ImageHash
	"""
	Convert a stored hash (hex, as retrieved from str(Imagehash))
	back to a Imagehash object. This method should be used for
	hashes generated by ImageHash up to version 3.7. For hashes
	generated by newer versions of ImageHash, hex_to_hash should
	be used instead.
	"""
	arr = []
	count = hash_size * (hash_size // 4)
	if len(hexstr) != count:
		emsg = 'Expected hex string size of {}.'
		raise ValueError(emsg.format(count))
	for i in range(count // 2):
		h = hexstr[i * 2:i * 2 + 2]
		v = int('0x' + h, 16)
		arr.append([v & 2**i > 0 for i in range(8)])
	return ImageHash(numpy.array(arr))


def average_hash(image, hash_size=8, mean=numpy.mean, frame=0, background=DEFAULT_BACKGROUND):
	# type: (Image.Image, int, MeanFunc, int, tuple) -> ImageHash
	"""
	Average Hash computation

	Implementation follows https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

	Step by step explanation: https://web.archive.org/web/20171112054354/https://www.safaribooksonline.com/blog/2013/11/26/image-hashing-with-python/ # noqa: E501

	@image must be a PIL instance.
	@mean how to determine the average luminescence. can try numpy.median instead.
	@frame which frame of a multi-frame GIF/APNG to hash (0 = first).
	@background opaque colour behind transparent pixels, white by default.

	The image is normalised by ``to_gray_image`` (first frame, EXIF
	orientation, alpha flattened onto white) before being resized to
	``hash_size * hash_size`` with Lanczos and compared against ``mean``.
	Pixels exactly equal to the mean are encoded as 0 (``>`` comparison).
	"""
	hash_size = _check_hash_size(hash_size)

	# Acquire the agreed visual input before any resampling.
	image = to_gray_image(image, frame=frame, background=background)

	# reduce size and complexity
	image = image.resize((hash_size, hash_size), ANTIALIAS)

	# find average pixel value; 'pixels' is an array of the pixel values, ranging from 0 (black) to 255 (white)
	pixels = numpy.asarray(image)
	avg = mean(pixels)

	# create string of bits
	diff = pixels > avg
	# make a hash
	return ImageHash(diff)


def phash(image, hash_size=8, highfreq_factor=4, frame=0, background=DEFAULT_BACKGROUND):
	# type: (Image.Image, int, int, int, tuple) -> ImageHash
	"""
	Perceptual Hash computation.

	Implementation follows https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

	@image must be a PIL instance.
	@frame which frame of a multi-frame GIF/APNG to hash (0 = first).
	@background opaque colour behind transparent pixels, white by default.

	The luminance image is resized to ``hash_size * highfreq_factor``, a 2-D DCT
	is taken with ``scipy.fftpack.dct`` along both axes, and the top-left
	``hash_size * hash_size`` block is compared with its median (ties -> 0).
	"""
	hash_size = _check_hash_size(hash_size)
	highfreq_factor = _check_int('highfreq_factor', highfreq_factor, 1)

	import scipy.fftpack
	img_size = hash_size * highfreq_factor
	image = to_gray_image(image, frame=frame, background=background).resize((img_size, img_size), ANTIALIAS)
	pixels = numpy.asarray(image)
	dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
	dctlowfreq = dct[:hash_size, :hash_size]
	med = numpy.median(dctlowfreq)
	diff = dctlowfreq > med
	return ImageHash(diff)


def phash_simple(image, hash_size=8, highfreq_factor=4, frame=0, background=DEFAULT_BACKGROUND):
	# type: (Image.Image, int, int, int, tuple) -> ImageHash
	"""
	Perceptual Hash computation.

	Implementation follows https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

	@image must be a PIL instance.
	"""
	hash_size = _check_hash_size(hash_size)
	highfreq_factor = _check_int('highfreq_factor', highfreq_factor, 1)
	import scipy.fftpack
	img_size = hash_size * highfreq_factor
	image = to_gray_image(image, frame=frame, background=background).resize((img_size, img_size), ANTIALIAS)
	pixels = numpy.asarray(image)
	dct = scipy.fftpack.dct(pixels)
	dctlowfreq = dct[:hash_size, 1:hash_size + 1]
	avg = dctlowfreq.mean()
	diff = dctlowfreq > avg
	return ImageHash(diff)


def dhash(image, hash_size=8, frame=0, background=DEFAULT_BACKGROUND):
	# type: (Image.Image, int, int, tuple) -> ImageHash
	"""
	Difference Hash computation.

	following https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html

	computes differences horizontally

	@image must be a PIL instance.
	@frame which frame of a multi-frame GIF/APNG to hash (0 = first).
	@background opaque colour behind transparent pixels, white by default.

	The luminance image is resized to ``(hash_size + 1) x hash_size`` and each
	bit compares a pixel with the one to its left (``left < right`` -> 1, ties
	-> 0).
	"""
	# resize(w, h), but numpy.array((h, w))
	hash_size = _check_hash_size(hash_size)

	image = to_gray_image(image, frame=frame, background=background).resize(
		(hash_size + 1, hash_size), ANTIALIAS)
	pixels = numpy.asarray(image)
	# compute differences between columns
	diff = pixels[:, 1:] > pixels[:, :-1]
	return ImageHash(diff)


def dhash_vertical(image, hash_size=8, frame=0, background=DEFAULT_BACKGROUND):
	# type: (Image.Image, int, int, tuple) -> ImageHash
	"""
	Difference Hash computation.

	following https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html

	computes differences vertically

	@image must be a PIL instance.
	@frame which frame of a multi-frame GIF/APNG to hash (0 = first).
	@background opaque colour behind transparent pixels, white by default.

	The luminance image is resized to ``hash_size x (hash_size + 1)`` and each
	bit compares a pixel with the one above it (``above < below`` -> 1, ties
	-> 0).
	"""
	# resize(w, h), but numpy.array((h, w))
	hash_size = _check_hash_size(hash_size)

	image = to_gray_image(image, frame=frame, background=background).resize(
		(hash_size, hash_size + 1), ANTIALIAS)
	pixels = numpy.asarray(image)
	# compute differences between rows
	diff = pixels[1:, :] > pixels[:-1, :]
	return ImageHash(diff)


def whash(
	image, hash_size=8, image_scale=None, mode='haar', remove_max_haar_ll=True, frame=0, background=DEFAULT_BACKGROUND
):
	# type: (Image.Image, int, int | None, WhashMode, bool, int, tuple) -> ImageHash
	"""
	Wavelet Hash computation.

	based on https://www.kaggle.com/c/avito-duplicate-ads-detection/

	@image must be a PIL instance.
	@hash_size must be a power of 2 and less than @image_scale.
	@image_scale must be power of 2 and less than image size. By default is equal to max
					power of 2 for an input image.
	@mode (see modes in pywt library):
					'haar' - Haar wavelets, by default
					'db4' - Daubechies wavelets
	@remove_max_haar_ll - remove the lowest low level (LL) frequency using Haar wavelet.

	The luminance image is resized to ``(image_scale, image_scale)``, scaled
	to 0..1 and wavelet-decomposed.  The final LL block is compared with its
	median (ties -> 0).
	"""
	import pywt
	if image_scale is not None:
		image_scale = _check_int('image_scale', image_scale, 2)
		if image_scale & (image_scale - 1) != 0:
			raise ImageHashSizeError('image_scale is not power of 2')
	hash_size = _check_hash_size(hash_size)
	if hash_size & (hash_size - 1) != 0:
		raise ImageHashSizeError('hash_size is not power of 2')

	image = to_gray_image(image, frame=frame, background=background)
	if image_scale is None:
		image_natural_scale = 2**int(numpy.log2(min(image.size)))
		image_scale = max(image_natural_scale, hash_size)

	ll_max_level = int(numpy.log2(image_scale))

	level = int(numpy.log2(hash_size))
	if level > ll_max_level:
		raise ImageHashSizeError('hash_size in a wrong range')
	dwt_level = ll_max_level - level

	image = image.resize((image_scale, image_scale), ANTIALIAS)
	pixels = numpy.asarray(image) / 255.

	# Remove low level frequency LL(max_ll) if @remove_max_haar_ll using haar filter
	if remove_max_haar_ll:
		coeffs = pywt.wavedec2(pixels, 'haar', level=ll_max_level)
		coeffs = list(coeffs)
		coeffs[0] *= 0
		pixels = pywt.waverec2(coeffs, 'haar')

	# Use LL(K) as freq, where K is log2(@hash_size)
	coeffs = pywt.wavedec2(pixels, mode, level=dwt_level)
	dwt_low = coeffs[0]
	# Non-orthogonal filters such as 'db4' pad the approximation block, so
	# trim it back to the agreed hash_size x hash_size grid.
	dwt_low = dwt_low[:hash_size, :hash_size]

	# Subtract median and compute hash
	med = numpy.median(dwt_low)
	diff = dwt_low > med
	return ImageHash(diff)


def colorhash(image, binbits=3, frame=0, background=DEFAULT_BACKGROUND):
	# type: (Image.Image, int, int, tuple) -> ImageHash
	"""
	Color Hash computation.

	Computes fractions of image in intensity, hue and saturation bins:

	* the first binbits encode the black fraction of the image
	* the next binbits encode the gray fraction of the remaining image (low saturation)
	* the next 6*binbits encode the fraction in 6 bins of saturation, for highly saturated parts of the remaining image
	* the next 6*binbits encode the fraction in 6 bins of saturation, for mildly saturated parts of the remaining image

	@binbits number of bits to use to encode each pixel fractions

	The visual RGB image (alpha flattened onto white, EXIF orientation
	applied, first frame) is converted to HSV with Pillow.  Saturation exactly
	``256 * 2 // 3`` belongs to the bright group (``>=``).
	"""
	if not isinstance(binbits, int) or isinstance(binbits, bool) or binbits < 1:
		raise ImageHashSizeError('binbits must be an integer >= 1')

	image = to_visual_image(image, frame=frame, background=background)
	# bin in hsv space:
	intensity = numpy.asarray(image.convert('L')).flatten()
	h, s, v = [numpy.asarray(v).flatten() for v in image.convert('HSV').split()]
	# black bin
	mask_black = intensity < 256 // 8
	frac_black = mask_black.mean()
	# gray bin (low saturation, but not black)
	mask_gray = s < 256 // 3
	frac_gray = numpy.logical_and(~mask_black, mask_gray).mean()
	# two color bins (medium and high saturation, not in the two above)
	mask_colors = numpy.logical_and(~mask_black, ~mask_gray)
	mask_faint_colors = numpy.logical_and(mask_colors, s < 256 * 2 // 3)
	mask_bright_colors = numpy.logical_and(mask_colors, s >= 256 * 2 // 3)

	c = max(1, mask_colors.sum())
	# in the color bins, make sub-bins by hue
	hue_bins = numpy.linspace(0, 255, 6 + 1)
	if mask_faint_colors.any():
		h_faint_counts, _ = numpy.histogram(h[mask_faint_colors], bins=hue_bins)
	else:
		h_faint_counts = numpy.zeros(len(hue_bins) - 1)
	if mask_bright_colors.any():
		h_bright_counts, _ = numpy.histogram(h[mask_bright_colors], bins=hue_bins)
	else:
		h_bright_counts = numpy.zeros(len(hue_bins) - 1)

	# now we have fractions in each category (6*2 + 2 = 14 bins)
	# convert to hash and discretize:
	maxvalue = 2**binbits
	values = [min(maxvalue - 1, int(frac_black * maxvalue)), min(maxvalue - 1, int(frac_gray * maxvalue))]
	for counts in list(h_faint_counts) + list(h_bright_counts):
		values.append(min(maxvalue - 1, int(counts / c * maxvalue)))
	# print(values)
	bitarray = []
	for v in values:
		bitarray += [v // (2**(binbits - i - 1)) % 2**(binbits - i) > 0 for i in range(binbits)]
	return ImageHash(numpy.asarray(bitarray).reshape((-1, binbits)))


class ImageMultiHash:
	"""
	This is an image hash containing a list of individual hashes for segments of the image.
	The matching logic is implemented as described in Efficient Cropping-Resistant Robust Image Hashing
	"""

	def __init__(self, hashes):
		# type: (list[ImageHash]) -> None
		self.segment_hashes = hashes  # type: list[ImageHash]

	def __eq__(self, other):
		# type: (object) -> bool
		if not isinstance(other, ImageMultiHash):
			return False
		if not self.segment_hashes or not other.segment_hashes:
			return not self.segment_hashes and not other.segment_hashes
		return self.matches(other)  # type: ignore

	def __ne__(self, other):
		# type: (object) -> bool
		return not self.__eq__(other)

	def __sub__(self, other, hamming_cutoff=None, bit_error_rate=None):
		# type: (ImageMultiHash, float | None, float | None) -> float
		matches, sum_distance = self.hash_diff(other, hamming_cutoff, bit_error_rate)
		max_difference = len(self.segment_hashes)
		if matches == 0:
			return max_difference
		max_distance = matches * len(self.segment_hashes[0])
		tie_breaker = 0 - (float(sum_distance) / max_distance)
		match_score = matches + tie_breaker
		return max_difference - match_score

	def __hash__(self):
		return hash(tuple(hash(segment) for segment in self.segment_hashes))

	def __str__(self):
		return ','.join(str(x) for x in self.segment_hashes)

	def __repr__(self):
		return repr(self.segment_hashes)

	def hash_diff(self, other_hash, hamming_cutoff=None, bit_error_rate=None):
		# type: (ImageMultiHash, float | None, float | None) -> tuple[int, int]
		"""
		Gets the difference between two multi-hashes, as a tuple. The first element of the tuple is the number of
		matching segments, and the second element is the sum of the hamming distances of matching hashes.
		NOTE: Do not order directly by this tuple, as higher is better for matches, and worse for hamming cutoff.
		:param other_hash: The image multi hash to compare against
		:param hamming_cutoff: The maximum hamming distance to a region hash in the target hash
		:param bit_error_rate: Percentage of bits which can be incorrect, an alternative to the hamming cutoff. The
		default of 0.25 means that the segment hashes can be up to 25% different
		"""
		# Set default hamming cutoff if it's not set.
		if hamming_cutoff is None:
			if bit_error_rate is None:
				bit_error_rate = 0.25
			hamming_cutoff = len(self.segment_hashes[0]) * bit_error_rate
		# Get the hash distance for each region hash within cutoff
		distances = []
		for segment_hash in self.segment_hashes:
			lowest_distance = min(
				segment_hash - other_segment_hash
				for other_segment_hash in other_hash.segment_hashes
			)
			if lowest_distance > hamming_cutoff:
				continue
			distances.append(lowest_distance)
		return len(distances), sum(distances)

	def matches(self, other_hash, region_cutoff=1, hamming_cutoff=None, bit_error_rate=None):
		# type: (ImageMultiHash, int, float | None, float | None) -> bool
		"""
		Checks whether this hash matches another crop resistant hash, `other_hash`.
		:param other_hash: The image multi hash to compare against
		:param region_cutoff: The minimum number of regions which must have a matching hash
		:param hamming_cutoff: The maximum hamming distance to a region hash in the target hash
		:param bit_error_rate: Percentage of bits which can be incorrect, an alternative to the hamming cutoff. The
		default of 0.25 means that the segment hashes can be up to 25% different
		"""
		matches, _ = self.hash_diff(other_hash, hamming_cutoff, bit_error_rate)
		return matches >= region_cutoff

	def best_match(self, other_hashes, hamming_cutoff=None, bit_error_rate=None):
		# type: (list[ImageMultiHash], float | None, float | None) -> ImageMultiHash
		"""
		Returns the hash in a list which is the best match to the current hash
		:param other_hashes: A list of image multi hashes to compare against
		:param hamming_cutoff: The maximum hamming distance to a region hash in the target hash
		:param bit_error_rate: Percentage of bits which can be incorrect, an alternative to the hamming cutoff.
		Defaults to 0.25 if unset, which means the hash can be 25% different
		"""
		return min(
			other_hashes,
			key=lambda other_hash: self.__sub__(other_hash, hamming_cutoff, bit_error_rate)
		)


def _find_region(remaining_pixels, segmented_pixels):
	"""
	Finds a region and returns a set of pixel coordinates for it.
	:param remaining_pixels: A numpy bool array, with True meaning the pixels are remaining to segment
	:param segmented_pixels: A set of pixel coordinates which have already been assigned to segment. This will be
	updated with the new pixels added to the returned segment.
	"""
	in_region = set()
	not_in_region = set()
	# Find the first pixel in remaining_pixels with a value of True
	available_pixels = numpy.transpose(numpy.nonzero(remaining_pixels))
	start = tuple(available_pixels[0])
	in_region.add(start)
	new_pixels = in_region.copy()
	while True:
		try_next = set()
		# Find surrounding pixels
		for pixel in new_pixels:
			x, y = pixel
			neighbours = [
				(x - 1, y),
				(x + 1, y),
				(x, y - 1),
				(x, y + 1)
			]
			try_next.update(neighbours)
		# Remove pixels we have already seen
		try_next.difference_update(segmented_pixels, not_in_region)
		# If there's no more pixels to try, the region is complete
		if not try_next:
			break
		# Empty new pixels set, so we know whose neighbour's to check next time
		new_pixels = set()
		# Check new pixels
		for pixel in try_next:
			if remaining_pixels[pixel]:
				in_region.add(pixel)
				new_pixels.add(pixel)
				segmented_pixels.add(pixel)
			else:
				not_in_region.add(pixel)
	return in_region


def _find_all_segments(pixels, segment_threshold, min_segment_size):
	"""
	Finds all the regions within an image pixel array, and returns a list of the regions.

	Note: Slightly different segmentations are produced when using pillow version 6 vs. >=7, due to a change in
	rounding in the greyscale conversion.
	:param pixels: A numpy array of the pixel brightnesses.
	:param segment_threshold: The brightness threshold to use when differentiating between hills and valleys.
	:param min_segment_size: The minimum number of pixels for a segment.
	"""
	img_width, img_height = pixels.shape
	# threshold pixels
	threshold_pixels = pixels > segment_threshold
	unassigned_pixels = numpy.full(pixels.shape, True, dtype=bool)

	segments = []
	already_segmented = set()

	# Add all the pixels around the border outside the image:
	already_segmented.update([(-1, z) for z in range(img_height)])
	already_segmented.update([(z, -1) for z in range(img_width)])
	already_segmented.update([(img_width, z) for z in range(img_height)])
	already_segmented.update([(z, img_height) for z in range(img_width)])

	# Find all the "hill" regions
	while numpy.bitwise_and(threshold_pixels, unassigned_pixels).any():
		remaining_pixels = numpy.bitwise_and(threshold_pixels, unassigned_pixels)
		segment = _find_region(remaining_pixels, already_segmented)
		# Apply segment
		if len(segment) > min_segment_size:
			segments.append(segment)
		for pix in segment:
			unassigned_pixels[pix] = False

	# Invert the threshold matrix, and find "valleys"
	threshold_pixels_i = numpy.invert(threshold_pixels)
	while len(already_segmented) < img_width * img_height:
		remaining_pixels = numpy.bitwise_and(threshold_pixels_i, unassigned_pixels)
		segment = _find_region(remaining_pixels, already_segmented)
		# Apply segment
		if len(segment) > min_segment_size:
			segments.append(segment)
		for pix in segment:
			unassigned_pixels[pix] = False

	return segments


def crop_resistant_hash(
	image,  # type: Image.Image
	hash_func=dhash,  # type: HashFunc
	limit_segments=None,  # type: int | None
	segment_threshold=128,  # type: int
	min_segment_size=500,  # type: int
	segmentation_image_size=300  # type: int
):
	# type: (...) -> ImageMultiHash
	"""
	Creates a CropResistantHash object, by the algorithm described in the paper "Efficient Cropping-Resistant Robust
	Image Hashing". DOI 10.1109/ARES.2014.85
	This algorithm partitions the image into bright and dark segments, using a watershed-like algorithm, and then does
	an image hash on each segment. This makes the image much more resistant to cropping than other algorithms, with
	the paper claiming resistance to up to 50% cropping, while most other algorithms stop at about 5% cropping.

	Note: Slightly different segmentations are produced when using pillow version 6 vs. >=7, due to a change in
	rounding in the greyscale conversion. This leads to a slightly different result.
	Segments whose bounding boxes round to zero width or height are skipped. If no segments remain, the whole
	image is hashed instead.
	:param image: The image to hash
	:param hash_func: The hashing function to use
	:param limit_segments: If you have storage requirements, you can limit to hashing only the M largest segments
	:param segment_threshold: Brightness threshold between hills and valleys. This should be static, putting it between
	peak and through dynamically breaks the matching
	:param min_segment_size: Minimum number of pixels for a hashable segment
	:param segmentation_image_size: Size which the image is resized to before segmentation
	"""
	segmentation_image_size = _check_int('segmentation_image_size', segmentation_image_size, 1)

	orig_image = to_visual_image(image)
	# Convert to gray scale and resize
	image = orig_image.convert('L').resize((segmentation_image_size, segmentation_image_size), ANTIALIAS)
	# Add filters
	image = image.filter(ImageFilter.GaussianBlur()).filter(ImageFilter.MedianFilter())
	pixels = numpy.array(image).astype(numpy.float32)

	segments = _find_all_segments(pixels, segment_threshold, min_segment_size)

	# If there are no segments, have 1 segment including the whole image
	if not segments:
		full_image_segment = {(0, 0), (segmentation_image_size - 1, segmentation_image_size - 1)}
		segments.append(full_image_segment)

	# If segment limit is set, discard the smaller segments
	if limit_segments:
		segments = sorted(segments, key=lambda s: len(s), reverse=True)[:limit_segments]

	# Create bounding box for each segment
	hashes = []
	for segment in segments:
		orig_w, orig_h = orig_image.size
		scale_w = float(orig_w) / segmentation_image_size
		scale_h = float(orig_h) / segmentation_image_size
		min_y = min(coord[0] for coord in segment) * scale_h
		min_x = min(coord[1] for coord in segment) * scale_w
		max_y = (max(coord[0] for coord in segment) + 1) * scale_h
		max_x = (max(coord[1] for coord in segment) + 1) * scale_w
		# Compute robust hash for each bounding box
		bounding_box = orig_image.crop((min_x, min_y, max_x, max_y))
		if 0 in bounding_box.size:
			continue
		hashes.append(hash_func(bounding_box))
		# Show bounding box
		# im_segment = image.copy()
		# for pix in segment:
		# 	im_segment.putpixel(pix[::-1], 255)
		# im_segment.show()
		# bounding_box.show()

	if not hashes:
		hashes.append(hash_func(orig_image))
	return ImageMultiHash(hashes)
