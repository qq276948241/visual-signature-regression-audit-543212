"""Offline fixture round-trips through every format Pillow can encode here.

There is no network access and no checked-in binary sample: each image is
generated from an explicit pixel array, encoded to an in-memory buffer with a
given format, reopened, and hashed.  Lossless formats must reproduce the
original bit map exactly; lossy formats are asserted only on what their
contract promises.  A format Pillow cannot encode in this environment fails
the test rather than being skipped, so a missing sample is visible.
"""

from __future__ import absolute_import, division, print_function

import unittest

import numpy

import imagehash

from . import fixtures
from . import report_format_coverage as coverage


def _bits(image_hash):
	return ''.join('1' if bit else '0' for bit in image_hash.hash.flatten())


LOSSLESS_ALGORITHMS = [
	('average_hash', imagehash.average_hash),
	('dhash', imagehash.dhash),
	('dhash_vertical', imagehash.dhash_vertical),
	('phash', imagehash.phash),
	('phash_simple', imagehash.phash_simple),
	('whash', imagehash.whash),
	('colorhash', imagehash.colorhash),
]

# (format, save kwargs) for formats whose pixel values survive a round trip.
# AVIF is lossy even at quality 100, so it lives in the lossy class below.
LOSSLESS_FORMATS = (
	('PNG', {}),
	('BMP', {}),
	('TIFF', {}),
	('PPM', {}),
	('TGA', {}),
	('WEBP', {'lossless': True}),
	('JPEG2000', {}),
	('DDS', {}),
	('DIB', {}),
	('IM', {}),
	('PCX', {}),
	('QOI', {}),
	('SGI', {}),
)


class FormatCoverageTest(unittest.TestCase):
	def test_required_formats_have_samples(self):
		# This is the "never skip a branch for a missing sample" guard: if a
		# required format cannot be encoded here the failure names it.
		exercised = coverage._exercised_formats()
		for fmt in coverage.REQUIRED_FORMATS:
			with self.subTest(fmt=fmt):
				self.assertEqual(exercised[fmt], 'ok', '{} has no working sample'.format(fmt))

	def test_report_script_succeeds(self):
		self.assertEqual(coverage.main(), 0)

	def test_report_names_pillow_version(self):
		import PIL
		import contextlib
		import io
		buffer = io.StringIO()
		with contextlib.redirect_stdout(buffer):
			coverage.main()
		output = buffer.getvalue()
		self.assertIn('Pillow {}'.format(PIL.__version__), output)


class LosslessFormatRoundTripTest(unittest.TestCase):
	def test_rgb_fixture_round_trips_through_every_lossless_format(self):
		image, array = fixtures.gradient_rgb(48, 48)
		for fmt, kwargs in LOSSLESS_FORMATS:
			with self.subTest(fmt=fmt):
				reopened = fixtures.encode(image, fmt, **kwargs)
				numpy.testing.assert_array_equal(numpy.asarray(reopened.convert('RGB')), array)
				for name, func in LOSSLESS_ALGORITHMS:
					self.assertEqual(_bits(func(reopened)), _bits(func(image)), name)

	def test_palette_fixture_round_trips(self):
		image, array = fixtures.quadrant_blocks(cell=8)
		expected = {name: _bits(func(image)) for name, func in LOSSLESS_ALGORITHMS}
		for fmt, kwargs in (('PNG', {}), ('GIF', {}), ('TIFF', {}), ('WEBP', {'lossless': True})):
			with self.subTest(fmt=fmt):
				reopened = fixtures.encode(image, fmt, **kwargs)
				for name, func in LOSSLESS_ALGORITHMS:
					self.assertEqual(_bits(func(reopened)), expected[name], name)

	def test_gray_fixture_round_trips(self):
		image, array = fixtures.half_split_gray()
		expected = {name: _bits(func(image)) for name, func in LOSSLESS_ALGORITHMS}
		for fmt, kwargs in (('PNG', {}), ('TIFF', {}), ('PPM', {}), ('BMP', {})):
			with self.subTest(fmt=fmt):
				reopened = fixtures.encode(image, fmt, **kwargs)
				for name, func in LOSSLESS_ALGORITHMS:
					self.assertEqual(_bits(func(reopened)), expected[name], name)


class LossyFormatTest(unittest.TestCase):
	def test_jpeg_gradient_keeps_distance_small(self):
		# A smooth gradient is the worst case for DCT quantisation, so assert
		# on the distance envelope rather than on identical bits.  phash is
		# the most sensitive because it works in the DCT domain itself.
		image, _ = fixtures.gradient_rgb(64, 64)
		reopened = fixtures.encode(image, 'JPEG', quality=100)
		for name, func in LOSSLESS_ALGORITHMS:
			with self.subTest(func=name):
				distance = func(image) - func(reopened)
				limit = 24 if name == 'phash' else 8
				self.assertLessEqual(distance, limit, '{} distance {}'.format(name, distance))

	def test_jpeg_solid_fixture_is_stable(self):
		# Solid quadrants survive JPEG quantization for the spatial
		# algorithms; colour drift of a couple of units can still flip a DCT
		# tie, so colours are checked on the coarse colour bins.
		image, _ = fixtures.quadrant_blocks(cell=16)
		reopened = fixtures.encode(image, 'JPEG', quality=95)
		spatial = (
			('average_hash', imagehash.average_hash),
			('dhash', imagehash.dhash),
			('dhash_vertical', imagehash.dhash_vertical),
			('whash', imagehash.whash),
		)
		for name, func in spatial:
			with self.subTest(func=name):
				self.assertEqual(_bits(func(reopened)), _bits(func(image)))
		self.assertLessEqual(imagehash.colorhash(image) - imagehash.colorhash(reopened), 2)

	def test_avif_lossy_distance_is_small(self):
		# AVIF stores chroma at reduced resolution, so the hue-sensitive
		# colorhash and the DCT-domain phash drift more than the spatial
		# algorithms.  Bounds are per-algorithm contracts, not a fudge: the
		# spatial hashes must be exact.
		image, _ = fixtures.gradient_rgb(48, 48)
		reopened = fixtures.encode(image, 'AVIF', quality=100, speed=10)
		self.assertEqual(_bits(imagehash.average_hash(reopened)), _bits(imagehash.average_hash(image)))
		self.assertEqual(_bits(imagehash.dhash(reopened)), _bits(imagehash.dhash(image)))
		self.assertEqual(_bits(imagehash.dhash_vertical(reopened)), _bits(imagehash.dhash_vertical(image)))
		self.assertEqual(_bits(imagehash.whash(reopened)), _bits(imagehash.whash(image)))
		self.assertLessEqual(imagehash.phash(image) - imagehash.phash(reopened), 32)
		self.assertLessEqual(imagehash.colorhash(image) - imagehash.colorhash(reopened), 4)


class MultiFrameFormatTest(unittest.TestCase):
	def test_animated_gif_round_trip(self):
		image, arrays = fixtures.animated_frames('GIF', [(255, 0, 0), (0, 255, 0), (0, 0, 255)])
		for index, array in enumerate(arrays):
			with self.subTest(frame=index):
				numpy.testing.assert_array_equal(
					numpy.asarray(imagehash.to_visual_image(image, frame=index)), array)

	def test_apng_round_trip(self):
		image, arrays = fixtures.animated_frames('PNG', [(255, 0, 0), (0, 255, 0), (0, 0, 255)])
		self.assertEqual(image.n_frames, 3)
		for index, array in enumerate(arrays):
			with self.subTest(frame=index):
				numpy.testing.assert_array_equal(
					numpy.asarray(imagehash.to_visual_image(image, frame=index)), array)

	def test_single_frame_gif_has_one_frame(self):
		image, array = fixtures.quadrant_blocks()
		reopened = fixtures.encode(image, 'GIF')
		self.assertEqual(imagehash._frame_count(reopened), 1)
		self.assertEqual(imagehash.to_visual_image(reopened).size, image.size)


if __name__ == '__main__':
	unittest.main()
