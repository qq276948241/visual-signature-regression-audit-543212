"""Exact bit-map contracts for every hash algorithm.

These tests assert full output bit strings, not just relative distances.  The
expected values were derived from hand-built pixel arrays and are documented
inline so a failure points at the specific resizing, conversion, frequency
selection, threshold or bit-order rule that changed.

Bit order convention shared by all algorithms: the hash array is flattened in
row-major (C) order and rendered most-significant bit first, i.e. bit *i* of
the hex string is ``hash[i // width, i % width]``.
"""

from __future__ import absolute_import, division, print_function

import unittest

import numpy
from PIL import Image

import imagehash

from . import fixtures


def _hsv_image(array):
	"""Build an RGB image from an HSV array without Pillow's deprecated mode kwarg."""
	channels = [Image.fromarray(array[..., index]) for index in range(3)]
	return Image.merge('HSV', channels).convert('RGB')


def _bits(image_hash):
	return ''.join('1' if bit else '0' for bit in image_hash.hash.flatten())


def _hex(bits):
	assert len(bits) % 4 == 0
	return '{:0{}x}'.format(int(bits, 2), len(bits) // 4)


class AverageHashBitMapTest(unittest.TestCase):
	def test_flat_image_has_no_bits(self):
		# Every pixel equals the mean, and the contract is ``pixel > mean``.
		image, _ = fixtures.flat_gray(128, size=8)
		self.assertEqual(_bits(imagehash.average_hash(image)), '0' * 64)

	def test_mean_tie_encodes_zero(self):
		# Mean of [[0, 75], [50, 175]] is exactly 75 -> two pixels equal the mean.
		image, array = fixtures.mean_tie_gray()
		result = imagehash.average_hash(image, hash_size=2)
		self.assertEqual(numpy.asarray(image).tolist(), array.tolist())
		self.assertEqual(_bits(result), '0001')
		self.assertEqual(str(result), '1')

	def test_axis_aligned_split(self):
		# Top row black, bottom row white after the 8x8 resize.
		image = Image.new('L', (8, 8), 0)
		image.paste(Image.new('L', (8, 4), 255), (0, 4))
		result = imagehash.average_hash(image)
		self.assertEqual(_bits(result), '00000000' * 4 + '11111111' * 4)
		self.assertEqual(str(result), '00000000ffffffff')

	def test_row_major_bit_order(self):
		# One white pixel in the top-left cell and one in the bottom-right.
		array = numpy.zeros((8, 8), dtype=numpy.uint8)
		array[0, 0] = 255
		array[7, 7] = 255
		result = imagehash.average_hash(Image.fromarray(array))
		bits = _bits(result)
		self.assertEqual(bits[0], '1', 'first pixel must be the first bit')
		self.assertEqual(bits[-1], '1', 'last pixel must be the last bit')
		self.assertEqual(bits[1:-1], '0' * 62)

	def test_quadrant_blocks(self):
		image, _ = fixtures.quadrant_blocks(cell=8)
		self.assertEqual(_bits(imagehash.average_hash(image)), '00000000' * 4 + '11111111' * 4)

	def test_hash_size_controls_bit_count(self):
		image, _ = fixtures.gradient_rgb(32, 32)
		for hash_size in (2, 4, 8, 16):
			with self.subTest(hash_size=hash_size):
				result = imagehash.average_hash(image, hash_size=hash_size)
				self.assertEqual(result.hash.size, hash_size * hash_size)
				self.assertEqual(len(_bits(result)), hash_size * hash_size)
				self.assertEqual(result.hash.shape, (hash_size, hash_size))

	def test_custom_mean_changes_threshold(self):
		# A mostly dark image with a bright block: the mean is pulled above
		# the bulk of the pixels while the median is not, so the two
		# thresholds must disagree on at least one cell.
		image = Image.new('L', (16, 16), 0)
		image.paste(Image.new('L', (4, 4), 255), (0, 0))
		by_mean = imagehash.average_hash(image, mean=numpy.mean)
		by_median = imagehash.average_hash(image, mean=numpy.median)
		self.assertNotEqual(_bits(by_mean), _bits(by_median))


class DifferenceHashBitMapTest(unittest.TestCase):
	def test_strictly_increasing_ramp(self):
		# 0, 28, ..., 224 with no uint8 wrap; the resize target equals the
		# source size, so all 64 neighbour comparisons are strictly increasing.
		image, array = fixtures.horizontal_ramp()
		numpy.testing.assert_array_equal(array[0], numpy.arange(9, dtype=numpy.uint8) * 28)
		result = imagehash.dhash(image)
		self.assertEqual(_bits(result), '1' * 64)
		self.assertEqual(str(result), 'f' * 16)

	def test_descending_ramp(self):
		image, array = fixtures.horizontal_ramp()
		flipped = Image.fromarray(array[:, ::-1].copy())
		self.assertEqual(_bits(imagehash.dhash(flipped)), '0' * 64)

	def test_equal_neighbours_are_zero(self):
		# Flat rows: left == right, and the comparison is strict ``>``.
		image = Image.new('L', (9, 8), 77)
		self.assertEqual(_bits(imagehash.dhash(image)), '0' * 64)

	def test_known_pattern(self):
		# 8x9 left-black/right-white split, resized to 9x8: the transition
		# lands between the middle columns, giving 0101... rows.
		image = Image.new('L', (8, 9), 0)
		image.paste(Image.new('L', (4, 9), 255), (4, 0))
		result = imagehash.dhash(image)
		self.assertEqual(_bits(result), '01011010' * 8)
		self.assertEqual(str(result), '5a5a5a5a5a5a5a5a')

	def test_vertical_variant_ramp(self):
		# 8 columns x 9 rows of 0, 28, ..., 224; every vertical neighbour is
		# strictly increasing, so all 64 bits are 1.
		image, _ = fixtures.vertical_ramp()
		result = imagehash.dhash_vertical(image)
		self.assertEqual(_bits(result), '1' * 64)
		self.assertEqual(str(result), 'f' * 16)

	def test_vertical_variant_split(self):
		# 8x9, rows 4..7 white, row 8 black. Only the 3->4 transition rises,
		# which is bit row 3 of the 8 output rows.
		image = Image.new('L', (8, 9), 0)
		image.paste(Image.new('L', (8, 4), 255), (0, 4))
		result = imagehash.dhash_vertical(image)
		self.assertEqual(_bits(result), '0' * 24 + '1' * 8 + '0' * 32)
		self.assertEqual(str(result), '000000ff00000000')

	def test_bit_order_matches_pixel_order(self):
		# Two-pixel-wide image: bit 0 is (row 0, col 0)->(row 0, col 1).
		array = numpy.array([[0, 255, 0], [255, 0, 255]], dtype=numpy.uint8)
		result = imagehash.dhash(Image.fromarray(array), hash_size=2)
		numpy.testing.assert_array_equal(numpy.asarray(result.hash), numpy.array([[True, False], [False, True]]))
		self.assertEqual(_bits(result), '1001')


class PerceptualHashBitMapTest(unittest.TestCase):
	def test_flat_dc_only_image(self):
		# A constant image has energy only in the DC coefficient, so exactly
		# one bit (the DC component at [0, 0]) is above the block median.
		for value in (0, 128, 255):
			with self.subTest(value=value):
				image = Image.new('L', (32, 32), value)
				bits = _bits(imagehash.phash(image))
				if value == 0:
					self.assertEqual(bits, '0' * 64)
				else:
					self.assertEqual(bits, '1' + '0' * 63)

	def test_dc_selection_is_top_left_block(self):
		# Black top half / white bottom half yields a strong vertical step.
		# Note the selected block is dct[:8, :8] (not the shifted
		# dct[:8, 1:9] used by phash_simple); only three cells are above the
		# median, and they are at (0,0), (3,0) and (7,0).
		image, _ = fixtures.half_split_gray()
		result = imagehash.phash(image)
		self.assertEqual(_bits(result), '10000000' + '00000000' * 2 + '10000000' + '00000000' * 3 + '10000000')
		self.assertEqual(str(result), '8000008000000080')
		shifted = imagehash.phash_simple(image)
		self.assertEqual(_bits(shifted), '0' * 64)
		self.assertEqual(str(shifted), '0000000000000000')

	def test_quadrant_blocks(self):
		image, _ = fixtures.quadrant_blocks(cell=8)
		self.assertEqual(_bits(imagehash.phash(image)), '1011001101001100101100111011001101001100010001101011000110011001')
		self.assertEqual(str(imagehash.phash(image)), 'b34cb3b34c46b199')

	def test_phash_simple_selects_shifted_block(self):
		# phash_simple skips the DC column: dct[:8, 1:9] instead of [:8, :8].
		image, _ = fixtures.quadrant_blocks(cell=8)
		self.assertEqual(_bits(imagehash.phash_simple(image)), '01110111' * 8)
		self.assertEqual(str(imagehash.phash_simple(image)), '7777777777777777')

	def test_checkerboard(self):
		# Alternating 4px cells: two checkerboard harmonics in the selected
		# block sit above the median, at rows 0 and 2 of the four 8-column bands.
		image, _ = fixtures.checkerboard()
		result = imagehash.phash(image)
		expected = '10000000' + ('01010101' + '00000000') * 3 + '01010101'
		self.assertEqual(_bits(result), expected)
		self.assertEqual(str(result), '8055005500550055')

	def test_hash_size_and_highfreq_factor(self):
		image, _ = fixtures.gradient_rgb(32, 32)
		for hash_size in (2, 4, 8):
			for factor in (2, 3, 4):
				with self.subTest(hash_size=hash_size, factor=factor):
					result = imagehash.phash(image, hash_size=hash_size, highfreq_factor=factor)
					self.assertEqual(result.hash.shape, (hash_size, hash_size))
					self.assertEqual(result.hash.size, hash_size * hash_size)


class WaveletHashBitMapTest(unittest.TestCase):
	def test_aligned_half_split_haar(self):
		# Top half dark -> the LL block's first four rows are below the median.
		image, _ = fixtures.half_split_gray(size=16)
		for remove_ll in (True, False):
			with self.subTest(remove_max_haar_ll=remove_ll):
				result = imagehash.whash(image, hash_size=8, image_scale=16, remove_max_haar_ll=remove_ll)
				self.assertEqual(_bits(result), '0' * 32 + '1' * 32)
				self.assertEqual(str(result), '00000000ffffffff')

	def test_aligned_vertical_split_haar(self):
		image, _ = fixtures.vertical_split_gray(size=16)
		result = imagehash.whash(image, hash_size=8, image_scale=16)
		self.assertEqual(_bits(result), '00001111' * 8)
		self.assertEqual(str(result), '0f0f0f0f0f0f0f0f')

	def test_checkerboard_haar(self):
		image, _ = fixtures.checkerboard(size=32, cell=4)
		result = imagehash.whash(image, hash_size=8, image_scale=32)
		self.assertEqual(_bits(result), '1010101001010101' * 4)

	def test_constant_image_semantics(self):
		# With remove_max_haar_ll the constant is removed and the block is
		# zero (ties -> 0). Without it, boundary effects make the median
		# differ from the value, so the result is not all zeros for 128.
		zero = Image.new('L', (16, 16), 0)
		self.assertEqual(_bits(imagehash.whash(zero, hash_size=8, image_scale=16)), '0' * 64)
		mid = Image.new('L', (16, 16), 128)
		self.assertEqual(_bits(imagehash.whash(mid, hash_size=8, image_scale=16)), '11101000' * 8)
		self.assertEqual(
			_bits(imagehash.whash(mid, hash_size=8, image_scale=16, remove_max_haar_ll=False)),
			'0' * 64,
		)

	def test_db4_bit_count_matches_hash_size(self):
		# Non-orthogonal filters pad the LL block; the contract trims it back
		# to hash_size x hash_size before thresholding.
		image, _ = fixtures.half_split_gray(size=16)
		result = imagehash.whash(image, hash_size=8, image_scale=16, mode='db4')
		self.assertEqual(result.hash.shape, (8, 8))
		self.assertEqual(result.hash.size, 64)
		self.assertEqual(_bits(result), '0' * 32 + '1' * 8 + '0' * 8 + '1' * 16)
		self.assertEqual(str(result), '00000000ff00ffff')

	def test_hash_size_always_yields_square_bit_count(self):
		image, _ = fixtures.gradient_rgb(64, 64)
		for mode in ('haar', 'db4'):
			for hash_size in (2, 4, 8, 16):
				with self.subTest(mode=mode, hash_size=hash_size):
					result = imagehash.whash(image, hash_size=hash_size, image_scale=64, mode=mode)
					self.assertEqual(result.hash.shape, (hash_size, hash_size))
					self.assertEqual(result.hash.size, hash_size * hash_size)
					self.assertEqual(len(_bits(result)), hash_size * hash_size)


class ColorHashBitMapTest(unittest.TestCase):
	def _uniform(self, color, size=(4, 4)):
		return Image.new('RGB', size, color)

	def test_pure_colours_binbits_1(self):
		# 14 bins: black, gray, faint-hue x6, bright-hue x6.
		cases = {
			'white': ('01000000000000', '1000'),
			'black': ('10000000000000', '2000'),
			'gray': ('01000000000000', '1000'),
			'red': ('00000000100000', '0020'),
			'green': ('00000000001000', '0008'),
		}
		for name, (expected_bits, expected_hex) in cases.items():
			with self.subTest(color=name):
				result = imagehash.colorhash(self._uniform(name), binbits=1)
				self.assertEqual(_bits(result), expected_bits)
				self.assertEqual(str(result), expected_hex)

	def test_pure_colours_binbits_3(self):
		cases = {
			'white': ('000111000000000000000000000000000000000000', '07000000000'),
			'black': ('111000000000000000000000000000000000000000', '38000000000'),
			'red': ('000000000000000000000000111000000000000000', '00000038000'),
			'green': ('000000000000000000000000000000111000000000', '00000000e00'),
		}
		for name, (expected_bits, expected_hex) in cases.items():
			with self.subTest(color=name):
				result = imagehash.colorhash(self._uniform(name), binbits=3)
				self.assertEqual(_bits(result), expected_bits)
				self.assertEqual(str(result), expected_hex)

	def test_half_red_half_black(self):
		array = numpy.zeros((4, 4, 3), dtype=numpy.uint8)
		array[:, :2] = (255, 0, 0)
		image = Image.fromarray(array)
		self.assertEqual(_bits(imagehash.colorhash(image, binbits=3)), '110000000000000000000000111000000000000000')
		self.assertEqual(str(imagehash.colorhash(image, binbits=3)), '30000038000')

	def test_black_threshold_is_exclusive(self):
		# mask_black is intensity < 256 // 8 == 32.
		below = imagehash.colorhash(self._uniform((31, 31, 31)), binbits=1)
		at = imagehash.colorhash(self._uniform((32, 32, 32)), binbits=1)
		self.assertEqual(_bits(below), '10000000000000')
		self.assertEqual(_bits(at), '01000000000000')

	def test_gray_threshold_is_exclusive(self):
		# mask_gray is saturation < 256 // 3 == 85. Pillow's RGB->HSV keeps the
		# saturation we build, so 84 is gray and 85 is already coloured.
		def saturated(s):
			array = numpy.zeros((4, 4, 3), dtype=numpy.uint8)
			array[..., 0] = 30
			array[..., 1] = s
			array[..., 2] = 255
			image = _hsv_image(array)
			actual = numpy.asarray(image.convert('HSV'))[..., 1]
			self.assertEqual(int(actual.min()), s)
			self.assertEqual(int(actual.max()), s)
			return image

		self.assertEqual(_bits(imagehash.colorhash(saturated(84), binbits=1))[:2], '01')
		self.assertEqual(_bits(imagehash.colorhash(saturated(85), binbits=1))[:2], '00')

	def test_saturation_upper_boundary_belongs_to_bright(self):
		# 256 * 2 // 3 == 170; the faint mask is ``<`` and bright is ``>=``.
		def uniform_hsv(saturation):
			array = numpy.zeros((16, 16, 3), dtype=numpy.uint8)
			array[..., 0] = 30
			array[..., 1] = saturation
			array[..., 2] = 255
			image = _hsv_image(array)
			round_tripped = numpy.asarray(image.convert('HSV'))[..., 1]
			self.assertEqual(int(round_tripped.min()), saturation)
			self.assertEqual(int(round_tripped.max()), saturation)
			return image

		below = imagehash.colorhash(uniform_hsv(169), binbits=1)
		at = imagehash.colorhash(uniform_hsv(170), binbits=1)
		above = imagehash.colorhash(uniform_hsv(171), binbits=1)
		self.assertEqual(at, above)
		self.assertNotEqual(at, below)

	def test_binbits_controls_bit_count(self):
		image = self._uniform('red')
		for binbits in (1, 2, 3, 4, 5):
			with self.subTest(binbits=binbits):
				result = imagehash.colorhash(image, binbits=binbits)
				self.assertEqual(result.hash.size, 14 * binbits)
				self.assertEqual(len(_bits(result)), 14 * binbits)

	def test_rotation_invariance(self):
		image, _ = fixtures.quadrant_blocks()
		self.assertEqual(
			_bits(imagehash.colorhash(image)),
			_bits(imagehash.colorhash(image.rotate(180))),
		)


class SimilarityBitMapTest(unittest.TestCase):
	def test_brightness_shift_preserves_bits(self):
		image, array = fixtures.checkerboard()
		shifted = Image.fromarray(numpy.clip(array.astype(numpy.int16) + 40, 0, 255).astype(numpy.uint8))
		self.assertEqual(_bits(imagehash.average_hash(image)), _bits(imagehash.average_hash(shifted)))
		self.assertEqual(_bits(imagehash.dhash(image)), _bits(imagehash.dhash(shifted)))

	def test_rotation_180_is_invariant(self):
		image, array = fixtures.checkerboard()
		flipped = Image.fromarray(array[::-1, ::-1].copy())
		for name, func in (
			('average_hash', imagehash.average_hash),
			('dhash', imagehash.dhash),
			('phash', imagehash.phash),
			('whash', imagehash.whash),
			('colorhash', imagehash.colorhash),
		):
			with self.subTest(func=name):
				self.assertEqual(_bits(func(image)), _bits(func(flipped)))

	def test_rotation_90_changes_bits_for_directional_algorithms(self):
		image, array = fixtures.checkerboard()
		rotated = Image.fromarray(numpy.rot90(array).copy())
		self.assertNotEqual(_bits(imagehash.average_hash(image)), _bits(imagehash.average_hash(rotated)))
		self.assertNotEqual(_bits(imagehash.dhash(image)), _bits(imagehash.dhash(rotated)))

	def test_transparent_edge_hashes_like_visible_image(self):
		transparent, _ = fixtures.transparent_edge_rgba()
		visible = Image.new('RGB', (8, 8), (255, 255, 255))
		visible.paste(Image.new('RGB', (4, 8), (0, 0, 0)), (0, 0))
		self.assertEqual(_bits(imagehash.average_hash(transparent)), _bits(imagehash.average_hash(visible)))
		self.assertEqual(_bits(imagehash.dhash(transparent)), _bits(imagehash.dhash(visible)))
		self.assertEqual(_bits(imagehash.phash(transparent)), _bits(imagehash.phash(visible)))


class ParameterValidationTest(unittest.TestCase):
	def test_hash_size_below_two_rejected(self):
		image = Image.new('RGB', (8, 8))
		for func in (imagehash.average_hash, imagehash.dhash, imagehash.dhash_vertical, imagehash.phash, imagehash.phash_simple):
			for size in (-1, 0, 1):
				with self.subTest(func=func.__name__, size=size):
					with self.assertRaises(ValueError):
						func(image, hash_size=size)
		for size in (-1, 0, 1):
			with self.subTest(func='whash', size=size):
				with self.assertRaises(ValueError):
					imagehash.whash(image, hash_size=size)

	def test_non_integer_sizes_rejected(self):
		image = Image.new('RGB', (8, 8))
		for value in ('8', 8.0, True, None):
			for func in (imagehash.average_hash, imagehash.dhash, imagehash.phash):
				with self.subTest(func=func.__name__, value=value):
					with self.assertRaises(ValueError):
						func(image, hash_size=value)

	def test_whash_power_of_two_validation(self):
		image = Image.new('RGB', (64, 64))
		with self.assertRaises(ValueError):
			imagehash.whash(image, hash_size=3)
		with self.assertRaises(ValueError):
			imagehash.whash(image, image_scale=5)
		with self.assertRaises(ValueError):
			imagehash.whash(image, hash_size=128, image_scale=64)

	def test_whash_accepts_assertion_error_contract(self):
		# Historically these were ``assert`` statements; keep that type too.
		image = Image.new('RGB', (64, 64))
		with self.assertRaises(AssertionError):
			imagehash.whash(image, hash_size=3)

	def test_colorhash_rejects_non_positive_binbits(self):
		image = Image.new('RGB', (4, 4))
		for value in (0, -1, -3):
			with self.subTest(value=value):
				with self.assertRaises(ValueError):
					imagehash.colorhash(image, binbits=value)

	def test_dhash_vertical_rejects_small_hash_size(self):
		image = Image.new('RGB', (8, 8))
		for size in (0, 1, -1):
			with self.subTest(size=size):
				with self.assertRaises(ValueError):
					imagehash.dhash_vertical(image, hash_size=size)

	def test_crop_resistant_rejects_invalid_segmentation_size(self):
		image = Image.new('RGB', (32, 32))
		for value in (0, -5, '300', 300.0, True):
			with self.subTest(value=value):
				with self.assertRaises(ValueError):
					imagehash.crop_resistant_hash(image, segmentation_image_size=value)


if __name__ == '__main__':
	unittest.main()
