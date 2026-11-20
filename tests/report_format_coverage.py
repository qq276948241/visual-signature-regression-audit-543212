#!/usr/bin/env python
"""Report the format coverage of the offline image fixtures.

Prints the Pillow version, the file formats Pillow can write in this
environment, which of them the test-suite exercises, and which ones cannot be
produced as fixtures.  Exits non-zero when a format that *should* be covered
is missing a sample, so a missing sample can never be mistaken for a pass and
no branch is skipped just because a sample is absent.

Usage::

	python3 tests/report_format_coverage.py
"""

from __future__ import absolute_import, division, print_function

import io
import os
import sys

import PIL
from PIL import Image, features

import numpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imagehash  # noqa: E402


#: Formats the fixture matrix promises to exercise (single-frame + animated).
REQUIRED_FORMATS = ('PNG', 'GIF', 'APNG', 'JPEG', 'BMP', 'TIFF', 'WEBP', 'PPM', 'TGA')
#: Formats used only for pixel-identical storage of test samples.
OPTIONAL_FORMATS = ('JPEG2000', 'AVIF', 'DDS', 'DIB', 'IM', 'PCX', 'QOI', 'SGI', 'MPO', 'ICNS')


def _writable_formats():
	Image.init()
	probe = Image.new('RGB', (8, 8), (10, 20, 30))
	writable = {}
	for fmt in sorted(set(Image.EXTENSION.values())):
		buffer = io.BytesIO()
		try:
			probe.save(buffer, format=fmt)
			buffer.seek(0)
			Image.open(buffer).load()
		except Exception as exc:
			writable[fmt] = type(exc).__name__
		else:
			writable[fmt] = None
	return writable


def _animated_frames_available(fmt):
	try:
		frames = [Image.new('RGB', (4, 4), color) for color in ((0, 0, 0), (255, 255, 255))]
		buffer = io.BytesIO()
		frames[0].save(buffer, format=fmt, save_all=True, append_images=frames[1:], duration=100, loop=0)
		buffer.seek(0)
		reopened = Image.open(buffer)
		return getattr(reopened, 'n_frames', 1) == len(frames)
	except Exception:
		return False


def _exercised_formats():
	exercised = {}
	probe = Image.new('RGB', (8, 8), (10, 20, 30))
	for fmt in REQUIRED_FORMATS + OPTIONAL_FORMATS:
		if fmt == 'APNG':
			exercised[fmt] = 'ok' if _animated_frames_available('PNG') else 'cannot encode multi-frame PNG'
			continue
		if fmt == 'GIF':
			status = _animated_frames_available('GIF')
		else:
			status = None
		buffer = io.BytesIO()
		try:
			probe.save(buffer, format=fmt)
			buffer.seek(0)
			Image.open(buffer).load()
		except Exception as exc:
			exercised[fmt] = 'cannot encode: {}'.format(type(exc).__name__)
			continue
		if status is False:
			exercised[fmt] = 'single frame only (animation unsupported)'
		else:
			exercised[fmt] = 'ok'
	return exercised


def _version_lines():
	lines = ['Pillow {}'.format(PIL.__version__), 'NumPy {}'.format(numpy.__version__)]
	for module_name in ('scipy', 'pywt'):
		try:
			module = __import__(module_name)
		except ImportError:
			reason = 'phash unavailable' if module_name == 'scipy' else 'whash unavailable'
			lines.append('{:<10s} MISSING ({})'.format(module_name, reason))
		else:
			lines.append('{:<10s} {}'.format(module_name, getattr(module, '__version__', 'unknown')))
	lines.append('imagehash {}'.format(imagehash.__version__))
	lines.append('')
	return lines


def _coverage_lines(exercised):
	lines = ['Fixture coverage (required):']
	failures = []
	for fmt in REQUIRED_FORMATS:
		status = exercised[fmt]
		if status == 'ok':
			lines.append('  {:<10s} covered'.format(fmt))
		else:
			lines.append('  {:<10s} NOT COVERED ({})'.format(fmt, status))
			failures.append(fmt)
	lines.append('')
	lines.append('Fixture coverage (optional):')
	for fmt in OPTIONAL_FORMATS:
		status = exercised[fmt]
		lines.append('  {:<10s} {}'.format(fmt, 'covered' if status == 'ok' else 'unavailable ({})'.format(status)))
	return lines, failures


def _feature_lines():
	lines = ['Optional format features:']
	for name in ('jpg', 'jpg_2000', 'zlib', 'libtiff', 'webp', 'freetype2'):
		try:
			available = features.check(name)
		except Exception:
			available = 'unknown'
		lines.append('  {:<10s} {}'.format(name, 'yes' if available is True else available))
	try:
		animated_webp = features.check('webp') and _animated_frames_available('WEBP')
	except Exception:
		animated_webp = 'unknown'
	lines.append('  {:<10s} {}'.format('webp_anim', 'yes' if animated_webp is True else animated_webp))
	return lines


def main():
	exercised = _exercised_formats()
	writable = _writable_formats()
	available = sorted(fmt for fmt, error in writable.items() if error is None)

	lines = _version_lines()
	coverage_lines, failures = _coverage_lines(exercised)
	lines.extend(coverage_lines)

	lines.append('')
	lines.append('Formats Pillow can write here: {}'.format(', '.join(available) or 'none'))
	uncovered = [fmt for fmt in available if fmt not in exercised]
	lines.append('Writable but not asserted by a dedicated fixture: {}'.format(
		', '.join(uncovered) or 'none'))

	lines.append('')
	lines.extend(_feature_lines())

	if failures:
		lines.append('')
		lines.append('ERROR: required formats without a sample: {}'.format(', '.join(failures)))

	print('\n'.join(lines))
	return 1 if failures else 0


if __name__ == '__main__':
	sys.exit(main())
