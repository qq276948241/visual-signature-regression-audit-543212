===========
ImageHash
===========

An image hashing library written in Python. ImageHash supports:

* Average hashing
* Perceptual hashing
* Difference hashing
* Wavelet hashing
* HSV color hashing (colorhash)
* Crop-resistant hashing

|CI|_ |Coveralls|_ 

Rationale
=========

Image hashes tell whether two images look nearly identical.
This is different from cryptographic hashing algorithms (like MD5, SHA-1)
where tiny changes in the image give completely different hashes. 
In image fingerprinting, we actually want our similar inputs to have
similar output hashes as well.

The image hash algorithms (average, perceptual, difference, wavelet)
analyse the image structure on luminance (without color information).
The color hash algorithm analyses the color distribution and 
black & gray fractions (without position information).

Installation
============

Based on PIL/Pillow Image, numpy and scipy.fftpack (for pHash)
Easy installation through `pypi`_::

	pip install imagehash

Basic usage
===========
::

	>>> from PIL import Image
	>>> import imagehash
	>>> hash = imagehash.average_hash(Image.open('tests/data/imagehash.png'))
	>>> print(hash)
	ffd7918181c9ffff
	>>> otherhash = imagehash.average_hash(Image.open('tests/data/peppers.png'))
	>>> print(otherhash)
	9f172786e71f1e00
	>>> print(hash == otherhash)
	False
	>>> print(hash - otherhash)  # hamming distance
	33

Each algorithm can also have its hash size adjusted (or in the case of
colorhash, its :code:`binbits`). Increasing the hash size allows an
algorithm to store more detail in its hash, increasing its sensitivity
to changes in detail.

Input preprocessing contract
============================

All public hash functions normalise their input the same way before
resizing or converting it (see :func:`imagehash.to_visual_image`):

* **Multi-frame images** (GIF/APNG) are read from a single frame.  The
  first frame is used by default; pass :code:`frame=` to pick another one.
  The caller's image is never seeked or otherwise modified.
* **EXIF orientation** is applied, so a rotated JPEG hashes like the image
  it visually represents.
* **Transparency** is flattened onto an opaque background (white by
  default; pass :code:`background=` to change it).  RGB values hidden
  behind alpha are never used.  Palette, ``L`` and ``RGB`` images that
  declare colour-key transparency in ``Image.info`` are treated the same
  way.
* Grayscale, palette, CMYK and other modes are converted to RGB, so the
  same picture stored in different modes hashes identically.

Empty images raise :class:`imagehash.ImageHashError`; invalid
``hash_size``/``binbits`` values raise
:class:`imagehash.ImageHashSizeError` (a :class:`ValueError`).  ``hash_size``
for :func:`imagehash.whash` must be a power of two; the resulting hash always
has exactly ``hash_size ** 2`` bits, including for the ``db4`` mode.

Reproducing the contract checks
================================

The contract above is pinned by offline, self-generated fixtures (no network
and no checked-in binaries); see ``tests/fixtures.py``.  Run::

	pytest

To print the Pillow/NumPy/SciPy/PyWavelets versions and the file-format
coverage of the fixtures — including any format that is missing a sample —::

	python3 tests/report_format_coverage.py

The script exits non-zero if a required format has no working sample, so a
missing fixture can never pass silently.


The demo script **find_similar_images** illustrates how to find similar
images in a directory.

Source hosted at GitHub: https://github.com/JohannesBuchner/imagehash

References
-----------

* Average hashing (`aHashref`_)
* Perceptual hashing (`pHashref`_)
* Difference hashing (`dHashref`_)
* Wavelet hashing (`wHashref`_)
* Crop-resistant hashing (`crop_resistant_hashref`_)

.. _aHashref: https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
.. _pHashref: https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
.. _dHashref: https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
.. _wHashref: https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5
.. _pypi: https://pypi.python.org/pypi/ImageHash
.. _crop_resistant_hashref: https://ieeexplore.ieee.org/document/6980335

Examples
=========

To help evaluate how different hashing algorithms behave, below are a few hashes applied
to two datasets. This will let you know what images an algorithm thinks are basically identical.

Example 1: Icon dataset
-----------------------

Source: 7441 free icons on GitHub (see examples/github-urls.txt).

The following pages show groups of images with the same hash (the hashing method sees them as the same).

* `phash <https://johannesbuchner.github.io/imagehash/index3.html>`__ (or `with z-transform <https://johannesbuchner.github.io/imagehash/index9.html>`__)
* `dhash <https://johannesbuchner.github.io/imagehash/index4.html>`__ (or `with z-transform <https://johannesbuchner.github.io/imagehash/index10.html>`__)
* `colorhash <https://johannesbuchner.github.io/imagehash/index7.html>`__
* `average_hash <https://johannesbuchner.github.io/imagehash/index2.html>`__ (`with z-transform <https://johannesbuchner.github.io/imagehash/index8.html>`__)

The hashes use hashsize=8; colorhash uses binbits=3.
You may want to adjust the hashsize or require some manhattan distance (hash1 - hash2 < threshold).

Example 2: Art dataset
----------------------

Source: 109259 art pieces from https://www.parismuseescollections.paris.fr/en/recherche/image-libre/.

The following pages show groups of images with the same hash (the hashing method sees them as the same).

* `phash <https://johannesbuchner.github.io/imagehash/art3.html>`__ (or `with z-transform <https://johannesbuchner.github.io/imagehash/art9.html>`__)
* `dhash <https://johannesbuchner.github.io/imagehash/art4.html>`__ (or `with z-transform <https://johannesbuchner.github.io/imagehash/art10.html>`__)
* `colorhash <https://johannesbuchner.github.io/imagehash/art7.html>`__
* `average_hash <https://johannesbuchner.github.io/imagehash/art2.html>`__ (`with z-transform <https://johannesbuchner.github.io/imagehash/art8.html>`__)

For understanding hash distances, check out these excellent blog posts:

* https://tech.okcupid.com/evaluating-perceptual-image-hashes-at-okcupid-e98a3e74aa3a
* https://content-blockchain.org/research/testing-different-image-hash-functions/

Storing hashes
==============

As illustrated above, hashes can be turned into strings.
The strings can be turned back into a ImageHash object as follows.

For single perceptual hashes::

	>>> original_hash = imagehash.phash(Image.open('tests/data/imagehash.png'))
	>>> hash_as_str = str(original_hash)
	>>> print(hash_as_str)
	ffd7918181c9ffff
	>>> restored_hash = imagehash.hex_to_hash(hash_as_str)
	>>> print(restored_hash)
	ffd7918181c9ffff
	>>> assert restored_hash == original_hash
	>>> assert str(restored_hash) == hash_as_str

For crop_resistant_hash::

	>>> original_hash = imagehash.crop_resistant_hash(Image.open('tests/data/imagehash.png'), min_segment_size=500, segmentation_image_size=1000)
	>>> hash_as_str = str(original_hash)
	>>> restored_hash = imagehash.hex_to_multihash(hash_as_str)
	>>> assert restored_hash == original_hash
	>>> assert str(restored_hash) == hash_as_str

For colorhash::

	>>> original_hash = imagehash.colorhash(Image.open('tests/data/imagehash.png'), binbits=3)
	>>> hash_as_str = str(original_hash)
	>>> restored_hash = imagehash.hex_to_flathash(hash_as_str, hashsize=3)

Efficient database search
-------------------------

For storing the hashes in a database and using fast hamming distance
searches, see pointers at https://github.com/JohannesBuchner/imagehash/issues/127
(a blog post on how to do this would be a great contribution!)

@KDJDEV points to https://github.com/KDJDEV/imagehash-reverse-image-search-tutorial and writes: 
In this tutorial I use PostgreSQL and `this extension <https://github.com/fake-name/pg-spgist_hamming>`_, 
and show how you can create a reverse image search using hashes generated by this library.


Changelog
----------

* 4.3: typing annotations by @Avasam @SpangleLabs and @nh2

* 4.2: Cropping-Resistant image hashing added by @SpangleLabs

* 4.1: Add examples and colorhash

* 4.0: Changed binary to hex implementation, because the previous one was broken for various hash sizes. This change breaks compatibility to previously stored hashes; to convert them from the old encoding, use the "old_hex_to_hash" function.

* 3.5: Image data handling speed-up

* 3.2: whash now also handles smaller-than-hash images

* 3.0: dhash had a bug: It computed pixel differences vertically, not horizontally.
       I modified it to follow `dHashref`_. The old function is available as dhash_vertical.

* 2.0: Added whash

* 1.0: Initial ahash, dhash, phash implementations.

Contributing
=============

Pull requests and new features are warmly welcome.

If you encounter a bug or have a question, please open a GitHub issue. You can also try Stack Overflow.

Other projects
==============

* https://github.com/commonsmachinery/blockhash-python
* https://github.com/acoomans/instagram-filters
* https://pippy360.github.io/transformationInvariantImageSearch/
* https://www.phash.org/
* https://pypi.org/project/dhash/
* https://github.com/thorn-oss/perception (based on imagehash code, depends on opencv)
* https://docs.opencv.org/3.4/d4/d93/group__img__hash.html

.. |CI| image:: https://github.com/JohannesBuchner/imagehash/actions/workflows/testing.yml/badge.svg
.. _CI: https://github.com/JohannesBuchner/imagehash/actions/workflows/testing.yml

.. |Coveralls| image:: https://coveralls.io/repos/github/JohannesBuchner/imagehash/badge.svg
.. _Coveralls: https://coveralls.io/github/JohannesBuchner/imagehash
