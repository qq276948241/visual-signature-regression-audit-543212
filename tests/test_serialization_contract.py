"""Serialisation and comparison contract for ``ImageHash``.

Covers hex <-> binary round trips, cross-length distance errors, equality and
approximate-threshold boundaries, and sorting of hashes parsed back from
strings.
"""

from __future__ import absolute_import, division, print_function

import unittest

import numpy

import imagehash


def _bits(image_hash):
	return ''.join('1' if bit else '0' for bit in image_hash.hash.flatten())


def _from_bits(bits):
	size = int(numpy.sqrt(len(bits)))
	assert size * size == len(bits)
	array = numpy.array([bool(int(bit)) for bit in bits]).reshape(size, size)
	return imagehash.ImageHash(array)


class HexBinaryRoundTripTest(unittest.TestCase):
	def test_str_hex_has_no_leading_zero_loss(self):
		# A hash whose most significant bits are zero must keep its width.
		hash_value = _from_bits('0' * 63 + '1')
		self.assertEqual(str(hash_value), '0000000000000001')
		self.assertEqual(len(str(hash_value)), 16)

	def test_binary_to_hex_known_values(self):
		cases = {
			'1': '1', '11': '3', '111': '7', '1111': 'f',
			'10000': '10', '110000': '30', '11110000': 'f0',
			'00001': '01', '000011': '03', '00001111': '0f',
			'10000001': '81', '11110000111100001111': 'f0f0f',
			'11110000000100100011010001010110011110001001101010111100110111101111': 'f0123456789abcdef',
		}
		for bits, expected in cases.items():
			with self.subTest(bits=bits):
				array = numpy.array([int(bit) for bit in bits])
				self.assertEqual(imagehash._binary_array_to_hex(array), expected)

	def test_hex_to_binary_known_values(self):
		cases = {
			'1': '0001', '2': '0010', '3': '0011', 'a': '1010', 'f': '1111',
			'101': '100000001', '1b1': '110110001', '0b1': '010110001',
			'f0f0': '1111000011110000', '0f0f': '0000111100001111',
			'000c': '0000000000001100', '100000d': '1000000000000000000001101',
		}
		for hexstr, expected in cases.items():
			with self.subTest(hexstr=hexstr):
				bits = ''.join('1' if bit else '0' for bit in imagehash.hex_to_hash(hexstr).hash.flatten())
				self.assertEqual(bits, expected)

	def test_exact_bit_array_round_trip(self):
		hexstr = '9e3cc1f0f0c3e7c0'
		expected = numpy.array([
			[True, False, False, True, True, True, True, False],
			[False, False, True, True, True, True, False, False],
			[True, True, False, False, False, False, False, True],
			[True, True, True, True, False, False, False, False],
			[True, True, True, True, False, False, False, False],
			[True, True, False, False, False, False, True, True],
			[True, True, True, False, False, True, True, True],
			[True, True, False, False, False, False, False, False],
		])
		parsed = imagehash.hex_to_hash(hexstr)
		numpy.testing.assert_array_equal(parsed.hash, expected)
		self.assertEqual(imagehash._binary_array_to_hex(expected), hexstr)
		self.assertEqual(str(parsed), hexstr)

	def test_square_hashes_round_trip_at_various_sizes(self):
		for hash_size in (2, 4, 8, 16):
			with self.subTest(hash_size=hash_size):
				rng = numpy.random.RandomState(hash_size)
				array = rng.randint(0, 2, size=(hash_size, hash_size)).astype(bool)
				original = imagehash.ImageHash(array)
				parsed = imagehash.hex_to_hash(str(original))
				numpy.testing.assert_array_equal(parsed.hash, array)
				self.assertEqual(str(parsed), str(original))
				self.assertEqual(parsed - original, 0)

	def test_flat_hash_round_trip_preserves_shape(self):
		# colorhash stores a (14, binbits) array; hex_to_flathash is its parser.
		for binbits in (1, 2, 3, 4):
			with self.subTest(binbits=binbits):
				rng = numpy.random.RandomState(binbits)
				array = rng.randint(0, 2, size=(14, binbits)).astype(bool)
				original = imagehash.ImageHash(array)
				parsed = imagehash.hex_to_flathash(str(original), 14)
				numpy.testing.assert_array_equal(parsed.hash, array)
				self.assertEqual(str(parsed), str(original))

	def test_multihash_round_trip(self):
		segments = [_from_bits('0' * 60 + format(i, '04b')) for i in range(1, 5)]
		multi = imagehash.ImageMultiHash(segments)
		parsed = imagehash.hex_to_multihash(str(multi))
		self.assertEqual(len(parsed.segment_hashes), len(segments))
		for original, round_tripped in zip(segments, parsed.segment_hashes):
			numpy.testing.assert_array_equal(original.hash, round_tripped.hash)
		self.assertEqual(str(parsed), str(multi))

	def test_hex_parse_is_case_insensitive(self):
		self.assertEqual(
			_bits(imagehash.hex_to_hash('00FF')),
			_bits(imagehash.hex_to_hash('00ff')),
		)

	def test_too_wide_hex_is_rejected(self):
		# 6 hex digits imply sqrt(24) -> a 4x4 (16-bit) square, so a value
		# needing >16 bits cannot be represented and must not be silently
		# truncated into a ragged array.
		with self.assertRaises(ValueError):
			imagehash.hex_to_hash('ABCDEF')
		with self.assertRaises(ValueError):
			imagehash.hex_to_hash('abc')
		with self.assertRaises(ValueError):
			imagehash.hex_to_hash('zz')

	def test_odd_hash_sizes_round_trip(self):
		# Sizes whose square is not a multiple of 4 need a spare nibble; the
		# parser uses the same floor(sqrt()) width as the serialiser.
		for hash_size in (3, 5, 7):
			with self.subTest(hash_size=hash_size):
				rng = numpy.random.RandomState(hash_size)
				array = rng.randint(0, 2, size=(hash_size, hash_size)).astype(bool)
				original = imagehash.ImageHash(array)
				parsed = imagehash.hex_to_hash(str(original))
				numpy.testing.assert_array_equal(parsed.hash, array)


class DistanceTest(unittest.TestCase):
	def test_identical_hashes_have_zero_distance(self):
		hash_value = _from_bits('1010' * 16)
		self.assertEqual(hash_value - _from_bits('1010' * 16), 0)

	def test_hamming_distance_counts_differing_bits(self):
		left = _from_bits('0000' * 16)
		right = _from_bits('1111' * 16)
		self.assertEqual(left - right, 64)
		mostly_same = _from_bits('0' * 62 + '11')
		self.assertEqual(left - mostly_same, 2)

	def test_distance_is_symmetric(self):
		left = _from_bits('1100' * 16)
		right = _from_bits('1010' * 16)
		self.assertEqual(left - right, right - left)

	def test_mismatched_lengths_raise(self):
		short = _from_bits('0' * 16)
		long = _from_bits('0' * 64)
		with self.assertRaises(TypeError):
			short - long
		with self.assertRaises(TypeError):
			long - short

	def test_mismatched_shapes_same_size_raise(self):
		# Subtraction compares shapes, not just bit counts, so a 64-bit
		# (14, 8) colour hash cannot be subtracted from an 8x8 binary hash.
		square = _from_bits('0' * 64)
		flat = imagehash.ImageHash(numpy.zeros((14, 8), dtype=bool))
		with self.assertRaises(TypeError):
			square - flat

	def test_distance_to_none_raises(self):
		with self.assertRaises(TypeError):
			_from_bits('0' * 64) - None

	def test_distance_to_non_hash_raises(self):
		hash_value = _from_bits('0' * 64)
		for value in (0, '0' * 64, numpy.zeros((8, 8), dtype=bool)):
			with self.subTest(value=type(value).__name__):
				with self.assertRaises(TypeError):
					hash_value - value


class EqualityTest(unittest.TestCase):
	def test_equal_hashes(self):
		self.assertTrue(_from_bits('1' * 64) == _from_bits('1' * 64))
		self.assertFalse(_from_bits('1' * 64) != _from_bits('1' * 64))

	def test_different_hashes(self):
		self.assertFalse(_from_bits('1' * 64) == _from_bits('0' * 64))
		self.assertTrue(_from_bits('1' * 64) != _from_bits('0' * 64))

	def test_non_hash_comparison_is_false_not_error(self):
		hash_value = _from_bits('1' * 64)
		for value in (0, 1, '1', None, object(), numpy.zeros((8, 8), dtype=bool)):
			with self.subTest(value=type(value).__name__):
				self.assertFalse(hash_value == value)
				self.assertTrue(hash_value != value)

	def test_hash_usable_as_dict_key(self):
		first = _from_bits('1' * 64)
		same = _from_bits('1' * 64)
		different = _from_bits('0' * 64)
		mapping = {first: 'value'}
		self.assertEqual(mapping[same], 'value')
		self.assertNotIn(different, mapping)

	def test_equality_after_serialisation(self):
		original = _from_bits('10110011' * 8)
		self.assertEqual(original, imagehash.hex_to_hash(str(original)))

	def test_multihash_equality_uses_matching(self):
		segments = [_from_bits('0' * 64), _from_bits('1' * 64)]
		left = imagehash.ImageMultiHash(segments)
		same = imagehash.ImageMultiHash([imagehash.ImageHash(s.hash.copy()) for s in segments])
		self.assertTrue(left == same)
		self.assertFalse(left == None)  # noqa: E711


class ThresholdBoundaryTest(unittest.TestCase):
	"""Comparison thresholds are inclusive on their documented side."""

	def _multi(self, hashes):
		return imagehash.ImageMultiHash(hashes)

	def test_hamming_cutoff_boundary(self):
		# 8x8 = 64 bits. A segment differing by exactly 16 bits is kept when
		# the cutoff is 16 and dropped when it is 15.
		base = _from_bits('0' * 64)
		differing = _from_bits('1' * 16 + '0' * 48)
		left = self._multi([base])
		right = self._multi([differing])
		self.assertEqual(left.hash_diff(right, hamming_cutoff=16), (1, 16))
		self.assertEqual(left.hash_diff(right, hamming_cutoff=15), (0, 0))

	def test_bit_error_rate_default_is_quarter(self):
		# 64-bit segments -> cutoff 16.0 by default.
		base = _from_bits('0' * 64)
		at = _from_bits('1' * 16 + '0' * 48)
		over = _from_bits('1' * 17 + '0' * 47)
		left = self._multi([base])
		self.assertEqual(left.hash_diff(self._multi([at])), (1, 16))
		self.assertEqual(left.hash_diff(self._multi([over])), (0, 0))

	def test_bit_error_rate_boundary(self):
		base = _from_bits('0' * 64)
		differing = _from_bits('1' * 18 + '0' * 46)
		left = self._multi([base])
		right = self._multi([differing])
		self.assertEqual(left.hash_diff(right, bit_error_rate=18 / 64), (1, 18))
		self.assertEqual(left.hash_diff(right, bit_error_rate=17.5 / 64), (0, 0))

	def test_matches_region_cutoff(self):
		segments = [_from_bits('0' * 64), _from_bits('1' * 64)]
		other = self._multi([_from_bits('0' * 64), _from_bits('0' * 64)])
		# Only the first segment is within the default cutoff; the second
		# differs by all 64 bits.
		self.assertTrue(self._multi(segments).matches(other, region_cutoff=1))
		self.assertFalse(self._multi(segments).matches(other, region_cutoff=2))
		self.assertFalse(self._multi(segments).matches(other, region_cutoff=3))

	def test_no_matches_returns_max_difference(self):
		left = self._multi([_from_bits('0' * 64), _from_bits('0' * 32 + '1' * 32)])
		right = self._multi([_from_bits('1' * 64)])
		self.assertEqual(left - right, len(left.segment_hashes))


class SortingTest(unittest.TestCase):
	def test_hashes_parsed_from_strings_sort(self):
		strings = ['ff00000000000000', '0f00000000000000', '000000000000000f', '0000000000000000']
		hashes = [imagehash.hex_to_hash(value) for value in strings]
		ordered = [str(value) for value in sorted(hashes)]
		self.assertEqual(ordered, sorted(strings))

	def test_sorting_is_consistent_with_equality(self):
		first = _from_bits('1010' * 16)
		same = imagehash.hex_to_hash(str(first))
		self.assertFalse(first < same)
		self.assertFalse(first > same)
		self.assertTrue(first <= same)
		self.assertTrue(first >= same)

	def test_sorting_mixed_sizes_is_total(self):
		# Ordering key starts with bit count, so a small hash sorts before a
		# large one even when its numeric value is bigger.
		small = _from_bits('1' * 4)
		large = _from_bits('0' * 64)
		ordered = sorted([large, small])
		self.assertEqual(str(ordered[0]), 'f')
		self.assertEqual(str(ordered[1]), '0' * 16)

	def test_multihash_string_parses_and_matches(self):
		segments = [_from_bits('0' * 64), _from_bits('1' * 64)]
		multi = imagehash.ImageMultiHash(segments)
		parsed = imagehash.hex_to_multihash(str(multi))
		self.assertEqual(str(parsed), str(multi))
		self.assertTrue(multi.matches(parsed, region_cutoff=1))


if __name__ == '__main__':
	unittest.main()
