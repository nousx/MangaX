"""Cached SHA-256 verification of local model files before they are loaded."""

import asyncio
import hashlib
import os
import tempfile
import unittest
from unittest import mock

import _bootstrap  # noqa: F401

from manga_translator.utils import inference, model_hash_cache
from manga_translator.utils.inference import ModelVerificationException, ModelWrapper

GOOD_CONTENT = b'model-weights' * 1000
GOOD_HASH = hashlib.sha256(GOOD_CONTENT).hexdigest()


class _DummyModel(ModelWrapper):
    _MODEL_SUB_DIR = 'dummy'
    _MODEL_MAPPING = {
        'model': {
            'url': 'https://example.invalid/weights.bin',
            'hash': GOOD_HASH,
            'file': 'weights.bin',
        },
        'no_hash': {
            'url': 'https://example.invalid/extra.bin',
            'file': 'extra.bin',
        },
    }

    def __init__(self):
        self.load_calls = 0
        super().__init__()

    async def _load(self, device: str, *args, **kwargs):
        self.load_calls += 1

    async def _unload(self):
        pass

    async def _infer(self, *args, **kwargs):
        pass


class ModelHashVerificationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.models_dir = os.path.join(self._tmp.name, 'models')
        self.cache_path = os.path.join(self._tmp.name, 'cache', 'model_hash_cache.json')
        os.makedirs(os.path.join(self.models_dir, 'dummy'))
        self.weights = os.path.join(self.models_dir, 'dummy', 'weights.bin')
        self._write(self.weights, GOOD_CONTENT)
        self._write(os.path.join(self.models_dir, 'dummy', 'extra.bin'), b'unhashed')

        env = mock.patch.dict(os.environ, {
            model_hash_cache.CACHE_PATH_ENV: self.cache_path,
            inference.ALLOW_UNVERIFIED_MODELS_ENV: '',
        })
        env.start()
        self.addCleanup(env.stop)
        models_dir = mock.patch.object(_DummyModel, '_MODEL_DIR', self.models_dir)
        models_dir.start()
        self.addCleanup(models_dir.stop)
        model_hash_cache.reset_memory_cache()
        self.addCleanup(model_hash_cache.reset_memory_cache)

    @staticmethod
    def _write(path, data):
        with open(path, 'wb') as f:
            f.write(data)

    def test_should_hash_once_then_use_cache(self):
        with mock.patch.object(model_hash_cache, 'sha256_file', wraps=model_hash_cache.sha256_file) as hasher:
            first = model_hash_cache.verify_file(self.weights, GOOD_HASH)
            second = model_hash_cache.verify_file(self.weights, GOOD_HASH.upper())
            model_hash_cache.reset_memory_cache()  # simulate a new process
            third = model_hash_cache.verify_file(self.weights, GOOD_HASH)

        self.assertEqual(first, (True, GOOD_HASH, False))
        self.assertEqual(second, (True, None, True))
        self.assertEqual(third, (True, None, True))
        self.assertEqual(hasher.call_count, 1)
        self.assertTrue(os.path.isfile(self.cache_path))

    def test_should_rehash_when_file_changes(self):
        model_hash_cache.verify_file(self.weights, GOOD_HASH)
        self._write(self.weights, b'tampered' + GOOD_CONTENT)

        ok, actual, from_cache = model_hash_cache.verify_file(self.weights, GOOD_HASH)

        self.assertFalse(ok)
        self.assertFalse(from_cache)
        self.assertNotEqual(actual, GOOD_HASH)

    def test_should_rehash_when_only_mtime_changes(self):
        model_hash_cache.verify_file(self.weights, GOOD_HASH)
        stat = os.stat(self.weights)
        os.utime(self.weights, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))

        ok, actual, from_cache = model_hash_cache.verify_file(self.weights, GOOD_HASH)

        self.assertEqual((ok, actual, from_cache), (True, GOOD_HASH, False))

    def test_should_not_cache_a_mismatch(self):
        self._write(self.weights, b'other content')

        first = model_hash_cache.verify_file(self.weights, GOOD_HASH)
        second = model_hash_cache.verify_file(self.weights, GOOD_HASH)

        self.assertFalse(first[0])
        self.assertFalse(second[0])
        self.assertFalse(second[2])

    def test_should_accept_text_file_that_differs_only_by_crlf_endings(self):
        lf_content = b'a\nb\nc\n'
        dictionary = os.path.join(self.models_dir, 'dummy', 'dictionary.txt')
        self._write(dictionary, lf_content.replace(b'\n', b'\r\n'))
        expected = hashlib.sha256(lf_content).hexdigest()

        first = model_hash_cache.verify_file(dictionary, expected)
        second = model_hash_cache.verify_file(dictionary, expected)

        self.assertEqual(first, (True, expected, False))
        self.assertEqual(second, (True, None, True))

    def test_should_reject_text_file_with_different_content(self):
        dictionary = os.path.join(self.models_dir, 'dummy', 'dictionary.txt')
        self._write(dictionary, b'a\r\nx\r\nc\r\n')

        ok, _, _ = model_hash_cache.verify_file(dictionary, hashlib.sha256(b'a\nb\nc\n').hexdigest())

        self.assertFalse(ok)

    def test_should_not_normalize_line_endings_of_model_weights(self):
        lf_content = b'a\nb\nc\n'
        self._write(self.weights, lf_content.replace(b'\n', b'\r\n'))

        ok, _, _ = model_hash_cache.verify_file(self.weights, hashlib.sha256(lf_content).hexdigest())

        self.assertFalse(ok)

    def test_should_survive_corrupt_cache_file(self):
        os.makedirs(os.path.dirname(self.cache_path))
        self._write(self.cache_path, b'{not json')

        ok, _, from_cache = model_hash_cache.verify_file(self.weights, GOOD_HASH)

        self.assertTrue(ok)
        self.assertFalse(from_cache)

    def test_should_load_model_when_hash_matches(self):
        model = _DummyModel()

        asyncio.run(model.load('cpu'))

        self.assertEqual(model.load_calls, 1)
        self.assertTrue(model.is_loaded())

    def test_should_refuse_to_load_model_when_hash_mismatches(self):
        self._write(self.weights, b'replaced checkpoint')
        model = _DummyModel()

        with self.assertRaises(ModelVerificationException) as ctx:
            asyncio.run(model.load('cpu'))

        self.assertIn('weights.bin', str(ctx.exception))
        self.assertEqual(model.load_calls, 0)
        self.assertFalse(model.is_loaded())

    def test_should_load_mismatching_model_when_explicitly_allowed(self):
        self._write(self.weights, b'replaced checkpoint')
        model = _DummyModel()

        with mock.patch.dict(os.environ, {inference.ALLOW_UNVERIFIED_MODELS_ENV: '1'}):
            asyncio.run(model.load('cpu'))

        self.assertEqual(model.load_calls, 1)

    def test_should_only_verify_file_entries_that_declare_a_hash(self):
        model = _DummyModel()

        entries = [(key, os.path.basename(path), digest) for key, path, digest in model._iter_hashed_model_files()]

        self.assertEqual(entries, [('model', 'weights.bin', GOOD_HASH)])


if __name__ == '__main__':
    unittest.main()
