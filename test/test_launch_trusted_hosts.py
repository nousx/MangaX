"""The installer must not disable TLS verification for HTTPS package indexes."""

import importlib.util
import os
import unittest
from unittest import mock

import _bootstrap

ROOT = _bootstrap.ROOT


def load_launch(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'packaging' / 'launch.py')
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TrustedHostArgsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launch = load_launch('launch_trusted_hosts_under_test')

    def setUp(self):
        env = mock.patch.dict(os.environ, {self.launch.PIP_TRUST_ALL_INDEX_HOSTS_ENV: ''})
        env.start()
        self.addCleanup(env.stop)

    def test_should_not_trust_https_indexes(self):
        urls = list(self.launch.MIRROR_URLS) + ['https://download.pytorch.org', None, '']

        self.assertEqual(self.launch.build_trusted_host_args(urls), '')

    def test_should_only_trust_plain_http_indexes(self):
        args = self.launch.build_trusted_host_args([
            'http://mirror.internal/simple/',
            'https://pypi.org/simple/',
            'http://mirror.internal/other/',
        ])

        self.assertEqual(args, ' --trusted-host mirror.internal')

    def test_should_restore_legacy_behaviour_when_opted_in(self):
        with mock.patch.dict(os.environ, {self.launch.PIP_TRUST_ALL_INDEX_HOSTS_ENV: '1'}):
            args = self.launch.build_trusted_host_args(['https://pypi.org/simple/'])

        self.assertEqual(args, ' --trusted-host pypi.org')


if __name__ == '__main__':
    unittest.main()
