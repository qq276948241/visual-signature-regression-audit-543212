"""Input-preprocessing contract for the public hash functions.

Every public function must see the same visual RGB input for the same
picture regardless of how it was stored: grayscale, palette, RGB, RGBA with
transparency, degenerate modes or a rotation tucked into EXIF.  These tests
pin that contract down with raw pixel arrays rather than with snapshots.
"""

from __future__ import absolute_import, division, print_function

import io
import pickle
import threading
import unittest

import numpy
from PIL import Image, ImageOps

import imagehash

from . import fixtures

HASH_FUNCS = [
	('average_hash', imagehash.average_hash),
	('dhash', imagehash.dhash),
	('dhash_vertical', imagehash.dhash_vertical),
	('phash', imagehash.phash),
	('phash_simple', imagehash.phash_simple),
	('whash', imagehash.whash),
	('colorhash', imagehash.colorhash),
]


def _bits(image_hash):
	return ''.join('1' if bit else '0' for bit in image_hash.hash.flatten())


def _rgb_array(image):
	return numpy.asarray(image.convert('RGB'))


class ToVisualImageTest(unittest.TestCase):
	def test_rgb_is_returned_as_rgb(self):
		image, array = fixtures.quadrant_blocks()
		visual = imagehash.to_visual_image(image)
		self.assertEqual(visual.mode, 'RGB')
		self.assertEqual(visual.size, image.size)
		numpy.testing.assert_array_equal(numpy.asarray(visual), array)

	def test_grayscale_expands_to_equal_channels(self):
		image, array = fixtures.half_split_gray()
		visual = imagehash.to_visual_image(image)
		expected = numpy.repeat(array[..., None], 3, axis=2)
		numpy.testing.assert_array_equal(numpy.asarray(visual), expected)

	def test_palette_uses_palette_colours(self):
		image, indices = fixtures.palette_image([(255, 0, 0), (0, 255, 0)], numpy.array([[0, 1], [1, 0]]))
		visual = imagehash.to_visual_image(image)
		numpy.testing.assert_array_equal(
			numpy.asarray(visual),
			numpy.array([[[255, 0, 0], [0, 255, 0]], [[0, 255, 0], [255, 0, 0]]], dtype=numpy.uint8),
		)

	def test_transparent_rgba_uses_declared_background(self):
		image, _ = fixtures.transparent_rgba((200, 10, 10), alpha=0)
		visual = imagehash.to_visual_image(image)
		numpy.testing.assert_array_equal(
			numpy.asarray(visual),
			numpy.full((8, 8, 3), 255, dtype=numpy.uint8),
		)

	def test_transparent_rgba_background_override(self):
		image, _ = fixtures.transparent_rgba((200, 10, 10), alpha=0)
		visual = imagehash.to_visual_image(image, background=(0, 0, 0))
		numpy.testing.assert_array_equal(
			numpy.asarray(visual),
			numpy.zeros((8, 8, 3), dtype=numpy.uint8),
		)
		named = imagehash.to_visual_image(image, background='black')
		numpy.testing.assert_array_equal(numpy.asarray(named), numpy.asarray(visual))

	def test_hidden_colour_behind_full_alpha_is_ignored(self):
		black_hidden, _ = fixtures.transparent_rgba((0, 0, 0), alpha=0)
		red_hidden, _ = fixtures.transparent_rgba((255, 0, 0), alpha=0)
		numpy.testing.assert_array_equal(
			numpy.asarray(imagehash.to_visual_image(black_hidden)),
			numpy.asarray(imagehash.to_visual_image(red_hidden)),
		)

	def test_semi_transparent_alpha_is_composited(self):
		image, array = fixtures.transparent_rgba((10, 20, 30), alpha=128)
		# Unpremultiply forces the stored RGB to the nearest representable
		# value for alpha 128, so derive the expectation from the round-tripped
		# foreground rather than from the raw array.
		foreground = numpy.asarray(image.convert('RGBA')).astype(numpy.int64)
		expected = (foreground[..., :3] * foreground[..., 3:4] + 255 * (255 - foreground[..., 3:4]) + 127) // 255
		visual = numpy.asarray(imagehash.to_visual_image(image)).astype(numpy.int64)
		numpy.testing.assert_array_equal(visual, expected)

	def test_opaque_alpha_matches_rgb(self):
		image, array = fixtures.transparent_edge_rgba()
		# The right half is fully transparent, so compare only the visible
		# left half against the corresponding opaque pixels.
		opaque = numpy.zeros((8, 8, 3), dtype=numpy.uint8)
		opaque[:, :4] = (0, 0, 0)
		numpy.testing.assert_array_equal(
			numpy.asarray(imagehash.to_visual_image(image))[:, :4],
			opaque[:, :4],
		)

	def test_la_mode_is_flattened(self):
		image, _ = fixtures.transparent_la(luminance=0, alpha=0)
		visual = imagehash.to_visual_image(image)
		numpy.testing.assert_array_equal(
			numpy.asarray(visual),
			numpy.full((8, 8, 3), 255, dtype=numpy.uint8),
		)

	def test_palette_transparency_is_flattened(self):
		image, _ = fixtures.palette_with_transparency()
		visual = imagehash.to_visual_image(image)
		expected = numpy.array([[[255, 0, 0], [255, 255, 255]], [[255, 255, 255], [255, 0, 0]]], dtype=numpy.uint8)
		numpy.testing.assert_array_equal(numpy.asarray(visual), expected)

	def test_premultiplied_la_is_expanded(self):
		image, _ = fixtures.premultiplied_la(luminance=10, alpha=128)
		# straight = round(10 * 255 / 128) = 20, composited over white:
		# round(20 * 128 / 255 + 255 * 127 / 255) = 137.
		visual = imagehash.to_visual_image(image)
		numpy.testing.assert_array_equal(
			numpy.asarray(visual),
			numpy.full((8, 8, 3), 137, dtype=numpy.uint8),
		)

	def test_premultiplied_la_matches_straight_la(self):
		for alpha, straight_luminance in ((0, 40), (1, 40), (64, 90), (128, 10), (255, 200)):
			with self.subTest(alpha=alpha):
				premultiplied = int(numpy.round(straight_luminance * alpha / 255.0))
				premul_image, _ = fixtures.premultiplied_la(luminance=premultiplied, alpha=alpha)
				straight_image, _ = fixtures.transparent_la(luminance=straight_luminance, alpha=alpha)
				numpy.testing.assert_array_equal(
					numpy.asarray(imagehash.to_visual_image(premul_image)),
					numpy.asarray(imagehash.to_visual_image(straight_image)),
				)

	def test_premultiplied_rgba_matches_straight_rgba(self):
		for alpha in (0, 64, 128, 200, 255):
			with self.subTest(alpha=alpha):
				straight = numpy.array([40, 80, 160], dtype=numpy.int64)
				premultiplied = tuple(int(v) for v in numpy.round(straight * alpha / 255.0))
				premul_image, _ = fixtures.premultiplied_rgba(premultiplied, alpha)
				straight_image, _ = fixtures.transparent_rgba(tuple(int(v) for v in straight), alpha)
				numpy.testing.assert_array_equal(
					numpy.asarray(imagehash.to_visual_image(premul_image)),
					numpy.asarray(imagehash.to_visual_image(straight_image)),
				)

	def test_exif_orientation_is_applied(self):
		image, _, orientation = fixtures.exif_oriented(size=(2, 3), orientation=6)
		self.assertEqual(image.size, (2, 3))
		visual = imagehash.to_visual_image(image)
		self.assertEqual(visual.size, (3, 2))
		self.assertEqual(image.size, (2, 3), 'the caller image must not be rotated in place')
		expect = numpy.asarray(ImageOps.exif_transpose(image).convert('RGB'))
		numpy.testing.assert_array_equal(numpy.asarray(visual), expect)

	def test_exif_orientation_round_trips_through_png(self):
		array = numpy.arange(64, dtype=numpy.uint8).reshape(8, 8)
		encoded = fixtures.encode_gray_png_with_exif(array, orientation=6)
		reopened = Image.open(io.BytesIO(encoded))
		visual = imagehash.to_visual_image(reopened)
		self.assertEqual(visual.size, (8, 8))
		# A 90-degree turn of the stored pixels is a transpose plus flips.
		expected = numpy.rot90(numpy.repeat(array[..., None], 3, axis=2), k=-1)
		numpy.testing.assert_array_equal(numpy.asarray(visual), expected)

	def test_missing_exif_leaves_geometry_alone(self):
		image, array = fixtures.quadrant_blocks()
		numpy.testing.assert_array_equal(numpy.asarray(imagehash.to_visual_image(image)), array)

	def test_empty_image_raises(self):
		for size in [(0, 0), (0, 5), (5, 0)]:
			with self.subTest(size=size):
				with self.assertRaises(imagehash.ImageHashError):
					imagehash.to_visual_image(Image.new('RGB', size))

	def test_non_image_raises_type_error(self):
		for value in (None, 42, 'tests/data/imagehash.png', numpy.zeros((4, 4))):
			with self.subTest(value=type(value).__name__):
				with self.assertRaises(TypeError):
					imagehash.to_visual_image(value)

	def test_invalid_background(self):
		image = Image.new('RGBA', (2, 2), (0, 0, 0, 0))
		for value in [(300, 0, 0), (1, 2), (1.5, 2, 3), 'definitely-not-a-colour']:
			with self.subTest(value=value):
				with self.assertRaises(ValueError):
					imagehash.to_visual_image(image, background=value)

	def test_caller_image_is_not_modified(self):
		image, array = fixtures.quadrant_blocks()
		before = numpy.asarray(image).copy()
		imagehash.to_visual_image(image)
		numpy.testing.assert_array_equal(numpy.asarray(image), before)
		numpy.testing.assert_array_equal(numpy.asarray(image), array)

	def test_pickle_round_trip_of_image_is_unaffected(self):
		image, array = fixtures.quadrant_blocks()
		original = pickle.dumps(numpy.asarray(image).tobytes())
		imagehash.to_visual_image(image)
		self.assertEqual(pickle.dumps(numpy.asarray(image).tobytes()), original)


class FrameSelectionTest(unittest.TestCase):
	def _assert_frames(self, fmt, expected_colors):
		image, arrays = fixtures.animated_frames(fmt, expected_colors)
		self.assertEqual(image.n_frames, len(expected_colors))
		for index, array in enumerate(arrays):
			with self.subTest(fmt=fmt, frame=index):
				visual = imagehash.to_visual_image(image, frame=index)
				numpy.testing.assert_array_equal(numpy.asarray(visual), array)
				self.assertEqual(image.tell(), 0, 'frame selection must not leave the caller seeked')

	def test_gif_frames_are_addressed_by_index(self):
		self._assert_frames('GIF', [(255, 0, 0), (0, 255, 0), (0, 0, 255)])

	def test_apng_frames_are_addressed_by_index(self):
		self._assert_frames('PNG', [(255, 0, 0), (0, 255, 0), (0, 0, 255)])

	def test_default_frame_is_first(self):
		for fmt in ('GIF', 'PNG'):
			image, arrays = fixtures.animated_frames(fmt, [(255, 0, 0), (0, 255, 0), (0, 0, 255)])
			with self.subTest(fmt=fmt):
				numpy.testing.assert_array_equal(numpy.asarray(imagehash.to_visual_image(image)), arrays[0])
				numpy.testing.assert_array_equal(numpy.asarray(imagehash.to_visual_image(image, frame=0)), arrays[0])

	def test_negative_frame_counts_from_end(self):
		for fmt in ('GIF', 'PNG'):
			image, arrays = fixtures.animated_frames(fmt, [(255, 0, 0), (0, 255, 0), (0, 0, 255)])
			with self.subTest(fmt=fmt):
				numpy.testing.assert_array_equal(numpy.asarray(imagehash.to_visual_image(image, frame=-1)), arrays[-1])

	def test_out_of_range_frame_raises(self):
		for fmt in ('GIF', 'PNG'):
			image, arrays = fixtures.animated_frames(fmt, [(255, 0, 0), (0, 255, 0)])
			for frame in (len(arrays), -len(arrays) - 1, 10 ** 6):
				with self.subTest(fmt=fmt, frame=frame):
					with self.assertRaises(imagehash.ImageHashError):
						imagehash.to_visual_image(image, frame=frame)

	def test_frame_must_be_int(self):
		image, _ = fixtures.animated_frames('GIF', [(0, 0, 0), (255, 255, 255)])
		for value in ('0', 1.0, True, None):
			with self.subTest(value=value):
				with self.assertRaises(TypeError):
					imagehash.to_visual_image(image, frame=value)

	def test_repeated_frame_reads_are_stable(self):
		for fmt in ('GIF', 'PNG'):
			image, arrays = fixtures.animated_frames(fmt, [(255, 0, 0), (0, 255, 0), (0, 0, 255)])
			with self.subTest(fmt=fmt):
				first = numpy.asarray(imagehash.to_visual_image(image))
				for _ in range(5):
					numpy.testing.assert_array_equal(numpy.asarray(imagehash.to_visual_image(image)), first)
					numpy.testing.assert_array_equal(
						numpy.asarray(imagehash.to_visual_image(image, frame=2)), arrays[2]
					)

	def test_caller_can_still_seek_afterwards(self):
		for fmt in ('GIF', 'PNG'):
			image, arrays = fixtures.animated_frames(fmt, [(255, 0, 0), (0, 255, 0), (0, 0, 255)])
			for index in range(len(arrays)):
				imagehash.to_visual_image(image, frame=index)
			with self.subTest(fmt=fmt):
				for index, array in enumerate(arrays):
					image.seek(index)
					self.assertEqual(image.convert('RGB').getpixel((0, 0)), tuple(int(v) for v in array[0, 0]))

	def test_select_frame_returns_detached_copy(self):
		image, arrays = fixtures.animated_frames('GIF', [(255, 0, 0), (0, 255, 0)])
		frame = imagehash.select_frame(image, frame=1)
		self.assertIsNot(frame, image)
		frame.putpixel((0, 0), (1, 2, 3))
		image.seek(1)
		self.assertEqual(image.convert('RGB').getpixel((0, 0)), (0, 255, 0))

	def test_select_frame_on_still_image_returns_copy(self):
		image, array = fixtures.quadrant_blocks()
		frame = imagehash.select_frame(image)
		self.assertIsNot(frame, image)
		numpy.testing.assert_array_equal(numpy.asarray(frame), numpy.asarray(image))

	def test_gif_transparency_does_not_peek_through(self):
		image = fixtures.transparent_gif_frames()
		self.assertIn('transparency', image.info)
		first = numpy.asarray(imagehash.to_visual_image(image, frame=0))
		numpy.testing.assert_array_equal(first, numpy.full((8, 8, 3), 255, dtype=numpy.uint8))

	def test_default_frame_is_independent_of_caller_position(self):
		# A caller that already seeked to frame 2 must still get frame 0 for
		# the default, and must be left on frame 2 afterwards.
		for fmt in ('GIF', 'PNG'):
			image, arrays = fixtures.animated_frames(fmt, [(255, 0, 0), (0, 255, 0), (0, 0, 255)])
			image.seek(2)
			with self.subTest(fmt=fmt):
				numpy.testing.assert_array_equal(numpy.asarray(imagehash.to_visual_image(image)), arrays[0])
				self.assertEqual(image.tell(), 2)
				self.assertEqual(image.convert('RGB').getpixel((0, 0)), (0, 0, 255))

	def test_in_place_fallback_restores_caller_frame(self):
		# When a fresh handle is impossible (e.g. an image already detached
		# from its file), the library seeks the caller's object and must put
		# it back exactly where it was.
		from unittest.mock import patch

		for fmt in ('GIF', 'PNG'):
			image, arrays = fixtures.animated_frames(fmt, [(255, 0, 0), (0, 255, 0), (0, 0, 255)])
			image.seek(2)
			with self.subTest(fmt=fmt), patch.object(imagehash, '_independent_handle', return_value=None):
				for index, array in enumerate(arrays):
					numpy.testing.assert_array_equal(
						numpy.asarray(imagehash.to_visual_image(image, frame=index)), array)
					self.assertEqual(image.tell(), 2)
				self.assertEqual(image.convert('RGB').getpixel((0, 0)), (0, 0, 255))


class PublicFunctionsShareTheContractTest(unittest.TestCase):
	def test_equivalent_storage_gives_identical_hashes(self):
		grayscale, array = fixtures.half_split_gray()
		variants = {
			'L': grayscale,
			'RGB': grayscale.convert('RGB'),
			'RGBA_opaque': grayscale.convert('RGBA'),
			'P': grayscale.convert('P'),
		}
		for name, func in HASH_FUNCS:
			expected = _bits(func(grayscale))
			for variant_name, variant in variants.items():
				with self.subTest(func=name, variant=variant_name):
					self.assertEqual(_bits(func(variant)), expected)

	def test_transparent_pixels_do_not_change_hash(self):
		for name, func in HASH_FUNCS:
			visual_with_black, _ = fixtures.transparent_rgba((0, 0, 0), alpha=0)
			visual_with_red, _ = fixtures.transparent_rgba((255, 0, 0), alpha=0)
			with self.subTest(func=name):
				self.assertEqual(_bits(func(visual_with_black)), _bits(func(visual_with_red)))

	def test_transparent_edge_matches_opaque_white(self):
		transparent, _ = fixtures.transparent_edge_rgba()
		opaque = Image.new('RGB', (8, 8), (255, 255, 255))
		opaque.paste(Image.new('RGB', (4, 8), (0, 0, 0)), (0, 0))
		for name, func in HASH_FUNCS:
			with self.subTest(func=name):
				self.assertEqual(_bits(func(transparent)), _bits(func(opaque)))

	def test_exif_orientation_matches_untagged_rotation(self):
		image, _, _ = fixtures.exif_oriented(size=(16, 16), orientation=6)
		rotated = ImageOps.exif_transpose(image)
		for name, func in HASH_FUNCS:
			with self.subTest(func=name):
				self.assertEqual(_bits(func(image)), _bits(func(rotated)))

	def test_public_functions_reject_empty_images(self):
		empty = Image.new('RGB', (0, 0))
		for name, func in HASH_FUNCS:
			with self.subTest(func=name):
				with self.assertRaises(imagehash.ImageHashError):
					func(empty)

	def test_public_functions_reject_non_images(self):
		for name, func in HASH_FUNCS:
			with self.subTest(func=name):
				with self.assertRaises(TypeError):
					func(None)

	def test_public_functions_do_not_modify_input(self):
		image, array = fixtures.quadrant_blocks()
		before = numpy.asarray(image).copy()
		for name, func in HASH_FUNCS:
			func(image)
		numpy.testing.assert_array_equal(numpy.asarray(image), before)
		numpy.testing.assert_array_equal(numpy.asarray(image), array)

	def test_frame_parameter_is_honoured_by_hashes(self):
		image, arrays = fixtures.animated_frames('GIF', [(0, 0, 0), (255, 255, 255)])
		black_hash = imagehash.average_hash(image, frame=0)
		white_hash = imagehash.average_hash(image, frame=1)
		self.assertEqual(_bits(black_hash), '0' * 64)
		self.assertEqual(_bits(white_hash), '0' * 64)  # a flat white frame has no contrast
		dhash_black = imagehash.dhash(image, frame=0)
		dhash_white = imagehash.dhash(image, frame=1)
		self.assertEqual(_bits(dhash_black), '0' * 64)
		self.assertEqual(_bits(dhash_white), '0' * 64)
		# Use colourhash, which does distinguish pure black from pure white.
		self.assertNotEqual(_bits(imagehash.colorhash(image, frame=0)), _bits(imagehash.colorhash(image, frame=1)))

	def test_repeated_calls_do_not_share_state(self):
		image, array = fixtures.gradient_rgb()
		for name, func in HASH_FUNCS:
			with self.subTest(func=name):
				first = func(image).hash.copy()
				for _ in range(3):
					numpy.testing.assert_array_equal(func(image).hash, first)

	def test_concurrent_hashing_is_deterministic(self):
		image, arrays = fixtures.animated_frames('GIF', [(255, 0, 0), (0, 255, 0), (0, 0, 255)])
		expected = [
			_bits(imagehash.average_hash(image, frame=index)) for index in range(len(arrays))
		]
		results = []
		errors = []
		barrier = threading.Barrier(8)

		def worker():
			try:
				barrier.wait()
				local = []
				for _ in range(20):
					local.append([_bits(imagehash.average_hash(image, frame=i)) for i in range(len(arrays))])
				results.append(local)
			except Exception as exc:  # pragma: no cover - surfaced below
				errors.append(exc)

		threads = [threading.Thread(target=worker) for _ in range(8)]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()
		self.assertEqual(errors, [])
		for local in results:
			for row in local:
				self.assertEqual(row, expected)


if __name__ == '__main__':
	unittest.main()
