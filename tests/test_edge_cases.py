"""Edge cases and defensive error paths of the preprocessing contract."""

from __future__ import absolute_import, division, print_function

import unittest
from unittest import mock

import numpy
from PIL import Image

import imagehash

from . import fixtures


class DuckTypedImageTest(unittest.TestCase):
	def test_object_with_mode_and_size_is_accepted(self):
		class FakeImage(object):
			mode = 'RGB'
			size = (2, 2)

		image = FakeImage()
		self.assertIs(imagehash._check_image(image), image)

	def test_objects_without_image_attributes_rejected(self):
		class NotAnImage(object):
			pass

		with self.assertRaises(TypeError):
			imagehash._check_image(NotAnImage())


class FrameCountFallbacksTest(unittest.TestCase):
	def test_missing_n_frames_counts_as_one(self):
		self.assertEqual(imagehash._frame_count(Image.new('RGB', (2, 2))), 1)

	def test_none_n_frames_counts_as_one(self):
		image = Image.new('RGB', (2, 2))
		with mock.patch.object(type(image), 'n_frames', new_callable=mock.PropertyMock, create=True, return_value=None):
			self.assertEqual(imagehash._frame_count(image), 1)

	def test_n_frames_that_raises_counts_as_one(self):
		image = Image.new('RGB', (2, 2))
		patched = mock.patch.object(
			type(image), 'n_frames', new_callable=mock.PropertyMock,
			create=True, side_effect=AttributeError('boom'),
		)
		with patched:
			self.assertEqual(imagehash._frame_count(image), 1)


class IndependentHandleFallbackTest(unittest.TestCase):
	def test_image_without_fp_falls_back_to_in_place(self):
		# An already-loaded still image has no readable ``fp`` in general; the
		# contract still works because a still image is copied directly.
		image, array = fixtures.quadrant_blocks()
		numpy.testing.assert_array_equal(
			numpy.asarray(imagehash.to_visual_image(image)), array)

	def test_unreadable_fp_falls_back(self):
		image = Image.new('RGB', (4, 4), 'white')

		class BrokenFp(object):
			def seek(self, *args):
				raise IOError('broken')

			def tell(self):
				return 0

			def read(self):
				return b''

		with mock.patch.object(type(image), 'fp', new_callable=mock.PropertyMock, create=True, return_value=BrokenFp()):
			self.assertIsNone(imagehash._independent_handle(image))

	def test_fp_without_seek_is_ignored(self):
		image = Image.new('RGB', (4, 4), 'white')
		with mock.patch.object(image, 'fp', object(), create=True):
			self.assertIsNone(imagehash._independent_handle(image))

	def test_copy_frame_in_place_survives_tell_and_restore_errors(self):
		image, arrays = fixtures.animated_frames('GIF', [(255, 0, 0), (0, 255, 0)])

		original_seek = image.seek
		state = {'seen': 0}

		def flaky_seek(index):
			# Fail only the restore back to frame 0 after the forward seek.
			state['seen'] += 1
			if state['seen'] == 2:
				raise OSError('cannot restore')
			return original_seek(index)

		with mock.patch.object(image, 'seek', flaky_seek):
			numpy.testing.assert_array_equal(
				numpy.asarray(imagehash._copy_frame_in_place(image, 1)), arrays[1])

	def test_empty_fp_read_falls_back(self):
		image = Image.new('RGB', (4, 4), 'white')

		class EmptyFp(object):
			def seek(self, *args):
				pass

			def tell(self):
				return 0

			def read(self):
				return b''

		with mock.patch.object(type(image), 'fp', new_callable=mock.PropertyMock, create=True, return_value=EmptyFp()):
			self.assertIsNone(imagehash._independent_handle(image))

	def test_multiframe_without_fp_reads_frame_zero(self):
		# Without a file pointer a fresh handle is impossible, but the
		# already-loaded frame still decodes.
		image, arrays = fixtures.animated_frames('GIF', [(255, 0, 0), (0, 255, 0)])
		with mock.patch.object(image, 'fp', None):
			numpy.testing.assert_array_equal(
				numpy.asarray(imagehash.to_visual_image(image)), arrays[0])

	def test_loaded_frame_can_still_be_copied_without_fp(self):
		image, arrays = fixtures.animated_frames('GIF', [(255, 0, 0), (0, 255, 0)])
		image.seek(1)
		image.load()
		with mock.patch.object(image, 'fp', None):
			numpy.testing.assert_array_equal(
				numpy.asarray(imagehash._copy_frame_in_place(image, 1)), arrays[1])

	def test_frame_count_that_cannot_be_read(self):
		image, arrays = fixtures.animated_frames('GIF', [(255, 0, 0), (0, 255, 0)])
		patched = mock.patch.object(
			type(image), 'n_frames', new_callable=mock.PropertyMock,
			create=True, side_effect=OSError('cannot read'),
		)
		with patched:
			self.assertEqual(imagehash._frame_count(image), 1)
			self.assertEqual(imagehash.to_visual_image(image).size, (8, 8))


class InvalidHexTest(unittest.TestCase):
	def test_hex_to_hash_rejects_non_hex(self):
		for value in ('zz', 'not-hex', '12g4', '', None):
			with self.subTest(value=value):
				with self.assertRaises(ValueError):
					imagehash.hex_to_hash(value)

	def test_hex_to_flathash_rejects_bad_hashsize(self):
		for value in (0, -1, 1.5, True, '14', None):
			with self.subTest(value=value):
				with self.assertRaises(ValueError):
					imagehash.hex_to_flathash('abc', value)

	def test_hex_to_flathash_rejects_non_hex(self):
		with self.assertRaises(ValueError):
			imagehash.hex_to_flathash('zz', 14)

	def test_hex_to_flathash_rejects_too_short(self):
		with self.assertRaises(ValueError):
			imagehash.hex_to_flathash('0', 14)

	def test_hex_to_flathash_rejects_too_wide(self):
		# 3 hex digits = 12 bits; 7 bins implies binbits 1 -> width 7.
		with self.assertRaises(ValueError):
			imagehash.hex_to_flathash('fff', 7)

	def test_negative_hex_is_rejected(self):
		with self.assertRaises(ValueError):
			imagehash.hex_to_hash('-ff')
		with self.assertRaises(ValueError):
			imagehash.hex_to_flathash('-ff', 14)

	def test_old_hex_to_hash_rejects_wrong_length(self):
		# hash_size 8 expects 8 * 2 = 16 hex characters.
		with self.assertRaises(ValueError):
			imagehash.old_hex_to_hash('ab', hash_size=8)

	def test_old_hex_to_hash_accepts_expected_length(self):
		# hash_size 4 -> count 4 * 1 = 4 hex chars -> 2 rows of 8 bits = 16.
		result = imagehash.old_hex_to_hash('abcd', hash_size=4)
		self.assertEqual(result.hash.size, 16)


class OrderingEdgeCaseTest(unittest.TestCase):
	def test_ordering_with_non_hash_returns_not_implemented(self):
		hash_value = imagehash.hex_to_hash('0000000000000000')
		for other in (0, '0', None, object()):
			for comparison in ('__lt__', '__le__', '__gt__', '__ge__'):
				with self.subTest(other=type(other).__name__, comparison=comparison):
					self.assertIs(getattr(hash_value, comparison)(other), NotImplemented)
			with self.assertRaises(TypeError):
				hash_value < other

	def test_multihash_repr_and_hash(self):
		segments = [imagehash.hex_to_hash('0000000000000000'), imagehash.hex_to_hash('ffffffffffffffff')]
		multi = imagehash.ImageMultiHash(segments)
		self.assertIsInstance(hash(multi), int)
		self.assertEqual(repr(multi), repr(segments))

	def test_empty_multihash_comparisons(self):
		empty = imagehash.ImageMultiHash([])
		self.assertTrue(empty == imagehash.ImageMultiHash([]))
		self.assertFalse(empty == imagehash.ImageMultiHash([imagehash.hex_to_hash('0000000000000000')]))


class BackgroundValidationTest(unittest.TestCase):
	def test_single_channel_and_bad_values(self):
		image = Image.new('RGBA', (2, 2), (0, 0, 0, 0))
		with self.assertRaises(TypeError):
			imagehash.to_visual_image(image, background=5)
		with self.assertRaises(ValueError):
			imagehash.to_visual_image(image, background=(1, 2, 3, 4))
		with self.assertRaises(ValueError):
			imagehash.to_visual_image(image, background=(1.0, 2, 3))
		with self.assertRaises(ValueError):
			imagehash.to_visual_image(image, background=(-1, 2, 3))

	def test_named_colour_equivalent_to_tuple(self):
		image = Image.new('RGBA', (2, 2), (0, 0, 0, 0))
		self.assertEqual(
			imagehash.to_visual_image(image, background='white').getpixel((0, 0)),
			imagehash.to_visual_image(image, background=(255, 255, 255)).getpixel((0, 0)),
		)


class NonImageByteInputTest(unittest.TestCase):
	def test_bytes_are_not_treated_as_images(self):
		encoded = fixtures.encode_gray_png(numpy.zeros((4, 4), dtype=numpy.uint8))
		with self.assertRaises(TypeError):
			imagehash.average_hash(encoded)


class CropResistantFallbackTest(unittest.TestCase):
	def test_flat_image_uses_whole_image_fallback(self):
		# No segment survives `min_segment_size`, so the whole-image segment
		# must be hashed rather than returning an empty multi-hash.
		image = Image.new('L', (64, 64), 128)
		result = imagehash.crop_resistant_hash(
			image, min_segment_size=64 * 64, segmentation_image_size=64)
		self.assertEqual(len(result.segment_hashes), 1)
		self.assertEqual(str(result), str(imagehash.dhash(image)))
		self.assertTrue(result.matches(result))

	def test_uniform_white_image_has_a_segment(self):
		image = Image.new('RGB', (64, 64), 'white')
		result = imagehash.crop_resistant_hash(
			image, min_segment_size=64 * 64, segmentation_image_size=64)
		self.assertEqual(len(result.segment_hashes), 1)


if __name__ == '__main__':
	unittest.main()
