"""Cross-algorithm, parameterised contract checks.

Every algorithm is exercised over the same fixture matrix so that a change in
one preprocessing rule (resize filter, grey conversion, alpha flattening,
frame selection) cannot silently diverge between algorithms.
"""

from __future__ import absolute_import, division, print_function

import threading
import unittest

import numpy
from PIL import Image

import imagehash

from . import fixtures


def _bits(image_hash):
	return ''.join('1' if bit else '0' for bit in image_hash.hash.flatten())


# (name, callable, expected bit-count function of the parameter)
ALGORITHMS = [
	('average_hash', imagehash.average_hash, lambda size: size * size),
	('dhash', imagehash.dhash, lambda size: size * size),
	('dhash_vertical', imagehash.dhash_vertical, lambda size: size * size),
	('phash', imagehash.phash, lambda size: size * size),
	('phash_simple', imagehash.phash_simple, lambda size: size * size),
	('whash', imagehash.whash, lambda size: size * size),
	('colorhash', imagehash.colorhash, lambda bins: 14 * bins),
]

# Fixtures whose visual content is well defined regardless of resampling.
FIXTURE_NAMES = [
	'gradient_rgb',
	'quadrant_blocks',
	'checkerboard',
	'half_split_gray',
	'vertical_split_gray',
]


def _fixture(name):
	return getattr(fixtures, name)()


class CrossAlgorithmMatrixTest(unittest.TestCase):
	def test_all_algorithms_produce_expected_bit_counts(self):
		image, _ = fixtures.gradient_rgb(32, 32)
		for name, func, expected in ALGORITHMS:
			parameter = 3 if name == 'colorhash' else 8
			with self.subTest(func=name):
				result = func(image, parameter)
				self.assertEqual(result.hash.size, expected(parameter))
				self.assertEqual(len(_bits(result)), expected(parameter))

	def test_all_algorithms_are_deterministic_across_fixtures(self):
		for fixture_name in FIXTURE_NAMES:
			image, _ = _fixture(fixture_name)
			for name, func, _ in ALGORITHMS:
				with self.subTest(fixture=fixture_name, func=name):
					first = _bits(func(image))
					self.assertEqual(_bits(func(image)), first)
					self.assertEqual(_bits(func(image.copy())), first)

	def test_opaque_rgba_agrees_for_every_algorithm(self):
		# Adding a fully opaque alpha channel is lossless for every algorithm,
		# including colorhash.
		for fixture_name in FIXTURE_NAMES:
			base, _ = _fixture(fixture_name)
			rgba = base.convert('RGB').convert('RGBA')
			for name, func, _ in ALGORITHMS:
				with self.subTest(fixture=fixture_name, func=name):
					self.assertEqual(_bits(func(rgba)), _bits(func(base.convert('RGB'))))

	def test_lossless_mode_variants_agree_per_algorithm(self):
		# L/P conversions are lossless only for fixtures that already have no
		# colour or use at most 256 distinct colours, so restrict them to
		# those.  colorhash additionally needs input without hue information.
		gray_fixtures = ('half_split_gray', 'vertical_split_gray')
		few_colour_fixtures = ('quadrant_blocks', 'checkerboard')
		for fixture_name in gray_fixtures:
			base, _ = _fixture(fixture_name)
			variants = (base.convert('RGB'), base.convert('RGBA'), base.convert('P'))
			for name, func, _ in ALGORITHMS:
				expected = _bits(func(base.convert('RGB')))
				for variant in variants:
					with self.subTest(fixture=fixture_name, func=name, variant=variant.mode):
						self.assertEqual(_bits(func(variant)), expected)
		luma_algorithms = [entry for entry in ALGORITHMS if entry[0] != 'colorhash']
		for fixture_name in few_colour_fixtures:
			base, _ = _fixture(fixture_name)
			palette = base.convert('RGB').convert('P')
			self.assertLessEqual(len(palette.getcolors(256) or []), 256)
			for name, func, _ in luma_algorithms:
				with self.subTest(fixture=fixture_name, func=name, variant='P'):
					self.assertEqual(_bits(func(palette)), _bits(func(base.convert('RGB'))))

	def test_background_choice_only_affects_transparent_pixels(self):
		image, _ = fixtures.transparent_edge_rgba()
		opaque = image.copy()
		opaque.paste((0, 0, 0, 255), (4, 0, 8, 8))
		# dhash_vertical sees the split as a horizontal edge, which neither
		# background choice changes, so only assert on the row/column
		# sensitive algorithms.
		sensitive_algorithms = [entry for entry in ALGORITHMS if entry[0] != 'dhash_vertical']
		for name, func, _ in sensitive_algorithms:
			with self.subTest(func=name):
				white = _bits(func(image, background=(255, 255, 255)))
				black = _bits(func(image, background=(0, 0, 0)))
				self.assertNotEqual(white, black)
		for name, func, _ in ALGORITHMS:
			with self.subTest(func=name, case='opaque'):
				# With fully opaque input the background must be irrelevant.
				self.assertEqual(
					_bits(func(opaque, background=(255, 255, 255))),
					_bits(func(opaque, background=(0, 0, 0))),
				)

	def test_hash_size_parameter_is_respected(self):
		image, _ = fixtures.gradient_rgb(64, 64)
		for name, func, expected in ALGORITHMS:
			if name == 'colorhash':
				sizes = (1, 2, 3)
			elif name == 'whash':
				# whash requires a power-of-two hash_size.  Use an explicit
				# image_scale so the small sizes stay valid.
				sizes = (2, 4, 8)
			else:
				sizes = (2, 3, 4, 8)
			for size in sizes:
				with self.subTest(func=name, size=size):
					if name == 'whash':
						result = func(image, size, image_scale=64)
					else:
						result = func(image, size)
					self.assertEqual(result.hash.size, expected(size))
					self.assertEqual(str(result), imagehash._binary_array_to_hex(result.hash.flatten()))

	def test_serialisation_round_trip_per_algorithm(self):
		image, _ = fixtures.quadrant_blocks()
		for name, func, _ in ALGORITHMS:
			with self.subTest(func=name):
				result = func(image)
				if name == 'colorhash':
					parsed = imagehash.hex_to_flathash(str(result), 14)
				else:
					parsed = imagehash.hex_to_hash(str(result))
				self.assertEqual(result, parsed)
				self.assertEqual(result - parsed, 0)

	def test_no_shared_buffers_between_calls(self):
		# Mutating one returned hash must not alter a later result.
		image, _ = fixtures.checkerboard()
		for name, func, _ in ALGORITHMS:
			with self.subTest(func=name):
				first = func(image)
				snapshot = first.hash.copy()
				first.hash[...] = ~first.hash
				second = func(image)
				numpy.testing.assert_array_equal(second.hash, snapshot)

	def test_concurrent_multialgorithm_hashing(self):
		image, _ = fixtures.gradient_rgb(32, 32)
		expected = {name: _bits(func(image)) for name, func, _ in ALGORITHMS}
		errors = []
		results = []
		barrier = threading.Barrier(6)

		def worker():
			try:
				barrier.wait()
				local = {}
				for _ in range(10):
					for name, func, _ in ALGORITHMS:
						local.setdefault(name, []).append(_bits(func(image)))
				results.append(local)
			except Exception as exc:  # pragma: no cover - reported below
				errors.append(exc)

		threads = [threading.Thread(target=worker) for _ in range(6)]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()
		self.assertEqual(errors, [])
		self.assertEqual(len(results), 6)
		for local in results:
			for name, values in local.items():
				self.assertEqual(set(values), {expected[name]}, name)


class ImageInputTypeMatrixTest(unittest.TestCase):
	"""Algorithms must accept every PIL mode Pillow can construct."""

	def _representable_modes(self):
		modes = {
			'1': Image.new('1', (16, 16), 1),
			'L': Image.new('L', (16, 16), 128),
			'LA': Image.new('LA', (16, 16), (128, 255)),
			'P': Image.new('P', (16, 16), 0),
			'RGB': Image.new('RGB', (16, 16), (10, 20, 30)),
			'RGBA': Image.new('RGBA', (16, 16), (10, 20, 30, 128)),
			'CMYK': Image.new('CMYK', (16, 16), (0, 0, 0, 0)),
			'YCbCr': Image.new('YCbCr', (16, 16), (0, 0, 0)),
			'I': Image.new('I', (16, 16), 1000),
			'F': Image.new('F', (16, 16), 100.0),
		}
		return modes

	def test_modes_convert_or_raise_clearly(self):
		for mode, image in self._representable_modes().items():
			for name, func, _ in ALGORITHMS:
				with self.subTest(mode=mode, func=name):
					try:
						result = func(image)
					except imagehash.ImageHashError:
						continue
					self.assertGreater(result.hash.size, 0)

	def test_grayscale_and_rgb_agree(self):
		array = numpy.arange(256, dtype=numpy.uint8).reshape(16, 16)
		gray = Image.fromarray(array)
		rgb = gray.convert('RGB')
		for name, func, _ in ALGORITHMS:
			with self.subTest(func=name):
				self.assertEqual(_bits(func(gray)), _bits(func(rgb)))

	def test_rgba_and_palette_do_not_use_hidden_colours(self):
		# Fully transparent pixels with three different hidden colours must
		# hash identically; this is the core "no silent background" rule.
		images = [fixtures.transparent_rgba(rgb, alpha=0)[0] for rgb in ((0, 0, 0), (255, 0, 0), (17, 200, 3))]
		for name, func, _ in ALGORITHMS:
			with self.subTest(func=name):
				hashes = {_bits(func(image)) for image in images}
				self.assertEqual(len(hashes), 1)


class CropResistantContractTest(unittest.TestCase):
	def test_crop_resistant_hash_uses_visual_input(self):
		image, _ = fixtures.gradient_rgb(64, 64)
		rgba = image.convert('RGBA')
		rgba.putalpha(255)
		self.assertEqual(
			str(imagehash.crop_resistant_hash(image)),
			str(imagehash.crop_resistant_hash(rgba)),
		)

	def test_crop_resistant_hash_serialises(self):
		image, _ = fixtures.checkerboard(size=64, cell=8)
		result = imagehash.crop_resistant_hash(image, min_segment_size=1, segmentation_image_size=32)
		self.assertGreater(len(result.segment_hashes), 0)
		self.assertEqual(str(imagehash.hex_to_multihash(str(result))), str(result))
		self.assertEqual(result, imagehash.hex_to_multihash(str(result)))

	def test_crop_resistant_hash_rejects_empty_image(self):
		with self.assertRaises(imagehash.ImageHashError):
			imagehash.crop_resistant_hash(Image.new('RGB', (0, 0)))


if __name__ == '__main__':
	unittest.main()
