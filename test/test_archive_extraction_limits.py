"""Archive-bomb protection of desktop_qt_ui.utils.archive_extractor."""

import io
import os
import tempfile
import unittest
import zipfile
from unittest import mock

import _bootstrap  # noqa: F401

from desktop_qt_ui.utils import archive_extractor
from desktop_qt_ui.utils.archive_extractor import (
    ArchiveLimitError,
    extract_images_from_archive,
    extract_images_from_cbz,
    extract_images_from_epub,
)


def _write_zip(path, members, compression=zipfile.ZIP_DEFLATED):
    with zipfile.ZipFile(path, 'w', compression) as zf:
        for name, data in members:
            zf.writestr(name, data)


class ArchiveExtractionLimitsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.out_dir = os.path.join(self.tmp, 'out')

    def _zip_path(self, name='chapter.cbz'):
        return os.path.join(self.tmp, name)

    def test_should_extract_images_in_natural_order_when_within_limits(self):
        path = self._zip_path()
        _write_zip(path, [
            ('ch/10.png', b'ten'),
            ('ch/2.png', b'two'),
            ('ch/readme.txt', b'not an image'),
            ('ch/1.jpg', b'one'),
        ])

        images = extract_images_from_cbz(path, self.out_dir)

        self.assertEqual(
            [os.path.basename(p) for p in images],
            ['0000_1.jpg', '0001_2.png', '0002_10.png'],
        )
        with open(images[2], 'rb') as f:
            self.assertEqual(f.read(), b'ten')

    def test_should_stream_members_larger_than_one_chunk(self):
        path = self._zip_path()
        payload = os.urandom(300_000)
        _write_zip(path, [('big.png', payload)])

        with mock.patch.object(archive_extractor, 'EXTRACT_CHUNK_SIZE', 4096):
            images = extract_images_from_cbz(path, self.out_dir)

        with open(images[0], 'rb') as f:
            self.assertEqual(f.read(), payload)

    def test_should_keep_members_inside_output_dir_when_names_traverse(self):
        path = self._zip_path()
        _write_zip(path, [('../../evil.png', b'x'), ('/abs/also.png', b'y')])

        images = extract_images_from_cbz(path, self.out_dir)

        out_real = os.path.realpath(self.out_dir)
        self.assertEqual(len(images), 2)
        for image in images:
            self.assertEqual(os.path.dirname(os.path.realpath(image)), out_real)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, 'evil.png')))

    def test_should_abort_when_image_count_exceeds_limit(self):
        path = self._zip_path()
        _write_zip(path, [(f'{i}.png', b'x') for i in range(6)])

        with mock.patch.object(archive_extractor, 'MAX_ARCHIVE_IMAGE_COUNT', 5):
            with self.assertRaises(ArchiveLimitError) as ctx:
                extract_images_from_cbz(path, self.out_dir)

        self.assertIn('chapter.cbz', str(ctx.exception))
        self.assertEqual(os.listdir(self.out_dir), [])

    def test_should_abort_when_entry_count_exceeds_limit(self):
        path = self._zip_path()
        _write_zip(path, [(f'{i}.txt', b'x') for i in range(6)])

        with mock.patch.object(archive_extractor, 'MAX_ARCHIVE_ENTRY_COUNT', 5):
            with self.assertRaises(ArchiveLimitError):
                extract_images_from_cbz(path, self.out_dir)

    def test_should_abort_when_single_member_exceeds_size_limit(self):
        path = self._zip_path()
        # Highly compressible: tiny on disk, large when inflated.
        _write_zip(path, [('ok.png', b'small'), ('bomb.png', b'\0' * 200_000)])
        self.assertLess(os.path.getsize(path), 5_000)

        with mock.patch.object(archive_extractor, 'MAX_ARCHIVE_MEMBER_SIZE', 100_000):
            with self.assertRaises(ArchiveLimitError):
                extract_images_from_cbz(path, self.out_dir)

        self.assertEqual(os.listdir(self.out_dir), [])

    def test_should_abort_when_total_size_exceeds_limit(self):
        path = self._zip_path()
        _write_zip(path, [(f'{i}.png', b'\0' * 40_000) for i in range(5)])

        with mock.patch.object(archive_extractor, 'MAX_ARCHIVE_TOTAL_SIZE', 100_000):
            with self.assertRaises(ArchiveLimitError):
                extract_images_from_cbz(path, self.out_dir)

        self.assertEqual(os.listdir(self.out_dir), [])

    def test_should_remove_partial_output_when_header_understates_size(self):
        # The declared size passes the pre-check; the limit must still be
        # enforced on the bytes actually decompressed.
        path = self._zip_path()
        _write_zip(path, [('0.png', b'\0' * 30_000), ('1.png', b'\0' * 30_000), ('2.png', b'\0' * 30_000)])

        real_check = archive_extractor._ExtractionBudget.check_declared

        def lenient_check(budget, declared_sizes):
            return real_check(budget, [0 for _ in declared_sizes])

        with mock.patch.object(archive_extractor, 'MAX_ARCHIVE_TOTAL_SIZE', 70_000), \
                mock.patch.object(archive_extractor, 'EXTRACT_CHUNK_SIZE', 8192), \
                mock.patch.object(archive_extractor._ExtractionBudget, 'check_declared', lenient_check):
            with self.assertRaises(ArchiveLimitError):
                extract_images_from_cbz(path, self.out_dir)

        # Two complete files and one partial file had been written; all are gone.
        self.assertEqual(os.listdir(self.out_dir), [])

    def test_should_not_leave_cache_dir_when_dispatcher_aborts(self):
        path = self._zip_path('volume.zip')
        _write_zip(path, [(f'{i}.png', b'x') for i in range(4)])

        with mock.patch.object(archive_extractor, 'MAX_ARCHIVE_IMAGE_COUNT', 3):
            with self.assertRaises(ArchiveLimitError):
                extract_images_from_archive(path, self.out_dir)

        self.assertFalse(os.path.exists(self.out_dir))

        # The same archive extracts normally once it is within the limits.
        images, out_dir = extract_images_from_archive(path, self.out_dir)
        self.assertEqual(len(images), 4)
        self.assertTrue(os.path.exists(os.path.join(out_dir, archive_extractor.EXTRACT_META_FILENAME)))

    def test_should_abort_epub_when_image_exceeds_size_limit(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('mimetype', 'application/epub+zip')
            zf.writestr('images/1.png', b'\0' * 200_000)
        path = os.path.join(self.tmp, 'book.epub')
        with open(path, 'wb') as f:
            f.write(buf.getvalue())

        with mock.patch.object(archive_extractor, 'MAX_ARCHIVE_MEMBER_SIZE', 100_000):
            with self.assertRaises(ArchiveLimitError):
                extract_images_from_epub(path, self.out_dir)

        self.assertEqual(os.listdir(self.out_dir), [])

    def test_should_have_defaults_suited_to_large_manga_volumes(self):
        self.assertGreaterEqual(archive_extractor.MAX_ARCHIVE_IMAGE_COUNT, 2_000)
        self.assertGreaterEqual(archive_extractor.MAX_ARCHIVE_MEMBER_SIZE, 100 * 1024 * 1024)
        self.assertGreaterEqual(archive_extractor.MAX_ARCHIVE_TOTAL_SIZE, 2 * 1024 * 1024 * 1024)


if __name__ == '__main__':
    unittest.main()
