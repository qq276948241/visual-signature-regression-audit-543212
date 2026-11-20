"""Self-contained, offline image fixtures with published raw pixels.

Every sample here is generated from explicit numpy arrays at import time, so
the tests never depend on a checked-in binary, a network download or Pillow's
resampling of a source photograph.  The helper functions return
``(pillow_image, raw_pixels)`` pairs where ``raw_pixels`` is the array the
expected bit map was derived from, letting tests assert on pixels and bit
order rather than on opaque snapshots.
"""

from __future__ import absolute_import, division, print_function

import io

import numpy

from PIL import Image


def _array_to_image(array, mode):
	"""Build a PIL image from an array, inferring the mode from its shape.

	``Image.fromarray(..., mode=...)`` is deprecated in recent Pillow, and the
	shape already determines the mode for the arrays used here.
	"""
	array = numpy.asarray(array, dtype=numpy.uint8)
	if array.ndim == 2:
		expected = 'L'
	elif array.shape[2] == 3:
		expected = 'RGB'
	elif array.shape[2] == 4:
		expected = 'RGBA'
	elif array.shape[2] == 2:
		expected = 'LA'
	else:
		raise ValueError('unsupported fixture array shape {!r}'.format(array.shape))
	if mode not in ('L', 'RGB', 'RGBA', 'LA'):
		raise ValueError('unsupported fixture mode {!r}'.format(mode))
	if expected != mode:
		raise ValueError('fixture mode {!r} does not match array shape {!r}'.format(mode, array.shape))
	return Image.fromarray(array)


def gradient_rgb(width=32, height=32):
	"""Smooth linear gradient: R rises left to right, G top to bottom.

	The image is deliberately larger than any default resize target so the
	Lanczos path is exercised on real interpolated values.
	"""
	x = numpy.arange(width, dtype=numpy.uint16)
	y = numpy.arange(height, dtype=numpy.uint16)
	red = numpy.tile((x * 255 // (width - 1)).astype(numpy.uint8), (height, 1))
	green = numpy.tile((y * 255 // (height - 1)).astype(numpy.uint8)[:, None], (1, width))
	blue = numpy.full((height, width), 64, dtype=numpy.uint8)
	array = numpy.dstack([red, green, blue])
	return _array_to_image(array, 'RGB'), array


def quadrant_blocks(cell=8):
	"""Four hard-edged quadrants: TL black, TR red, BL green, BR white.

	Hard edges survive Lanczos resizing, so the expected bits are stable and
	can be written out by hand.
	"""
	size = cell * 2
	array = numpy.zeros((size, size, 3), dtype=numpy.uint8)
	array[:cell, cell:] = (255, 0, 0)
	array[cell:, :cell] = (0, 255, 0)
	array[cell:, cell:] = (255, 255, 255)
	return _array_to_image(array, 'RGB'), array


def half_split_gray(value=255, size=16):
	"""Top half black, bottom half ``value``; aligned to any power-of-two grid."""
	array = numpy.zeros((size, size), dtype=numpy.uint8)
	array[size // 2:, :] = value
	return _array_to_image(array, 'L'), array


def vertical_split_gray(size=16):
	"""Left half black, right half white."""
	array = numpy.zeros((size, size), dtype=numpy.uint8)
	array[:, size // 2:] = 255
	return _array_to_image(array, 'L'), array


def horizontal_ramp(width=9, height=8, step=28):
	"""Strictly increasing row (0, 28, ..., 224) so every difference bit is 1.

	``step=28`` keeps the ninth sample at 224, below the uint8 wrap point.
	"""
	row = (numpy.arange(width, dtype=numpy.uint16) * step).astype(numpy.uint8)
	array = numpy.tile(row, (height, 1))
	return _array_to_image(array, 'L'), array


def vertical_ramp(width=8, height=9, step=28):
	"""Strictly increasing column for ``dhash_vertical`` (no uint8 wrap)."""
	column = (numpy.arange(height, dtype=numpy.uint16) * step).astype(numpy.uint8)
	array = numpy.tile(column[:, None], (1, width))
	return _array_to_image(array, 'L'), array


def checkerboard(size=32, cell=4):
	"""Alternating black/white cells; 180-degree rotation is invariant."""
	array = numpy.zeros((size, size), dtype=numpy.uint8)
	yy, xx = numpy.mgrid[0:size, 0:size]
	array[(xx // cell + yy // cell) % 2 == 0] = 255
	return _array_to_image(array, 'L'), array


def flat_gray(value=128, size=8):
	array = numpy.full((size, size), value, dtype=numpy.uint8)
	return _array_to_image(array, 'L'), array


def mean_tie_gray():
	"""2x2 image whose mean (75) intersects two pixel values exactly.

	The average-hash contract is ``pixel > mean``, so pixels equal to the mean
	produce 0 bits.
	"""
	array = numpy.array([[0, 75], [50, 175]], dtype=numpy.uint8)
	return _array_to_image(array, 'L'), array


def transparent_rgba(rgb, alpha, size=(8, 8)):
	"""Uniform RGBA image with an explicit colour hidden behind alpha."""
	array = numpy.zeros((size[1], size[0], 4), dtype=numpy.uint8)
	array[..., :3] = rgb
	array[..., 3] = alpha
	return _array_to_image(array, 'RGBA'), array


def transparent_edge_rgba():
	"""Opaque left half, fully transparent right half with a red hidden colour.

	The hidden red must not leak into the luminance-related hashes; the
	visible result is black | white after compositing over white.
	"""
	array = numpy.zeros((8, 8, 4), dtype=numpy.uint8)
	array[:, :4] = (0, 0, 0, 255)
	array[:, 4:] = (255, 0, 0, 0)
	return _array_to_image(array, 'RGBA'), array


def transparent_la(luminance=0, alpha=0):
	array = numpy.zeros((8, 8, 2), dtype=numpy.uint8)
	array[..., 0] = luminance
	array[..., 1] = alpha
	return _array_to_image(array, 'LA'), array


def palette_image(colors, indices):
	"""Palette image built from ``colors`` (list of RGB) and index array."""
	height, width = indices.shape
	image = Image.new('P', (width, height), 0)
	palette = []
	for color in colors:
		palette.extend(color)
	palette.extend([0] * (256 * 3 - len(palette)))
	image.putpalette(palette)
	image.putdata([int(v) for v in indices.flatten()])
	return image, numpy.asarray(indices, dtype=numpy.uint8)


def palette_with_transparency():
	"""Palette image where index 1 is declared transparent via ``info``."""
	indices = numpy.array([[0, 1], [1, 0]], dtype=numpy.uint8)
	image, array = palette_image([(255, 0, 0), (0, 255, 0)], indices)
	image.info['transparency'] = 1
	return image, array


def exif_oriented(rgb=(10, 20, 30), size=(2, 3), orientation=6):
	"""Still image carrying an EXIF orientation tag without rotating pixels.

	Orientation 6 means "rotate 90 degrees clockwise for display", so the
	visual image has swapped width and height.
	"""
	array = numpy.zeros((size[1], size[0], 3), dtype=numpy.uint8)
	array[..., :] = rgb
	image = _array_to_image(array, 'RGB')
	exif = image.getexif()
	exif[0x0112] = orientation
	image.info['exif'] = exif.tobytes()
	return image, array, orientation


def encode(image, fmt, **kwargs):
	"""Serialise ``image`` to bytes in ``fmt`` and reopen it."""
	buffer = io.BytesIO()
	image.save(buffer, format=fmt, **kwargs)
	buffer.seek(0)
	return Image.open(buffer)


def animated_frames(fmt, colors, size=(8, 8), **kwargs):
	"""Multi-frame GIF/APNG whose frames have solid, distinct colours.

	Returns ``(opened_image, [per-frame arrays])``.  The per-frame arrays are
	the exact RGB values the test expects ``select_frame`` to yield.
	"""
	frames = []
	arrays = []
	for color in colors:
		array = numpy.zeros((size[1], size[0], 3), dtype=numpy.uint8)
		array[..., :] = color
		arrays.append(array)
		frames.append(_array_to_image(array, 'RGB'))
	buffer = io.BytesIO()
	frames[0].save(
		buffer,
		format=fmt,
		save_all=True,
		append_images=frames[1:],
		duration=100,
		loop=0,
		**kwargs
	)
	buffer.seek(0)
	return Image.open(buffer), arrays


def transparent_gif_frames(size=(8, 8)):
	"""Two-frame GIF with a declared transparent palette index.

	The transparent colour is pure red so a naive ``convert('L')`` would show
	it as luminance 76 instead of the white background the contract promises.
	"""
	palette = [255, 0, 0, 0, 0, 255] + [0, 0, 0] * 254
	frames = []
	for index in (0, 1):
		frame = Image.new('P', size, index)
		frame.putpalette(palette)
		frames.append(frame)
	buffer = io.BytesIO()
	frames[0].save(
		buffer,
		format='GIF',
		save_all=True,
		append_images=frames[1:],
		duration=100,
		loop=0,
		transparency=0,
		disposal=2,
	)
	buffer.seek(0)
	return Image.open(buffer)


def premultiplied_la(luminance=10, alpha=128, size=(8, 8)):
	"""Premultiplied ``La`` image built from raw bytes.

	Pillow cannot convert ``La`` to anything else, so it must be expanded by
	the library itself.
	"""
	row = bytes([luminance, alpha]) * size[0]
	image = Image.frombytes('La', size, row * size[1])
	return image, numpy.array([luminance, alpha], dtype=numpy.uint8)


def premultiplied_rgba(rgb, alpha, size=(8, 8)):
	"""Premultiplied ``RGBa`` image; ``rgb`` values are already multiplied."""
	row = bytes(list(rgb) + [alpha]) * size[0]
	image = Image.frombytes('RGBa', size, row * size[1])
	return image, numpy.array(list(rgb) + [alpha], dtype=numpy.uint8)


def encode_gray_png(array):
	buffer = io.BytesIO()
	_array_to_image(array, 'L').save(buffer, format='PNG')
	buffer.seek(0)
	return buffer.getvalue()


def encode_gray_png_with_exif(array, orientation):
	image = _array_to_image(array, 'L')
	exif = image.getexif()
	exif[0x0112] = orientation
	buffer = io.BytesIO()
	image.save(buffer, format='PNG', exif=exif)
	buffer.seek(0)
	return buffer.getvalue()
