"""Choosing the dependency variant: one function per kind of graphics card."""
import _bootstrap  # noqa: F401, I001

import importlib.util
from types import SimpleNamespace

import pytest

ROOT = _bootstrap.ROOT


@pytest.fixture
def launch():
    spec = importlib.util.spec_from_file_location("launch_variant_choice_test", ROOT / "packaging" / "launch.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def answer(monkeypatch, *replies):
    remaining = list(replies)
    monkeypatch.setattr("builtins.input", lambda prompt="": remaining.pop(0))
    return remaining


@pytest.mark.parametrize("reply, expected", [("", "cuda13.0"), ("y", "cuda13.0"), ("n", "cpu")])
def test_nvidia_should_offer_the_variant_the_driver_supports(launch, monkeypatch, reply, expected):
    answer(monkeypatch, reply)

    assert launch._variant_for_nvidia("NVIDIA GeForce RTX 4070", 13, "13.0", (8, 9)) == expected


def test_nvidia_should_ask_again_after_an_invalid_reply(launch, monkeypatch):
    remaining = answer(monkeypatch, "maybe", "n")

    assert launch._variant_for_nvidia("NVIDIA GeForce RTX 4070", 13, "13.0", (8, 9)) == "cpu"
    assert remaining == []


def test_nvidia_should_fall_back_to_cuda_12_6_when_the_version_is_unknown(launch, monkeypatch):
    answer(monkeypatch, "")

    assert launch._variant_for_nvidia("NVIDIA GeForce GTX 1660", None, None, None) == "cuda12.6"


def test_nvidia_should_only_offer_cpu_when_the_driver_is_too_old(launch, monkeypatch):
    answer(monkeypatch, "")

    assert launch._variant_for_nvidia("NVIDIA GeForce GTX 1660", 11, "11.8", (7, 5)) == "cpu"


def test_apple_silicon_should_use_metal_without_asking(launch, monkeypatch):
    answer(monkeypatch)

    assert launch._variant_for_apple_silicon("Apple M2") == "metal"


@pytest.mark.parametrize("reply, expected", [("1", "cuda13.0"), ("", "cpu"), ("2", "cpu")])
def test_other_gpu_should_default_to_cpu(launch, monkeypatch, reply, expected):
    answer(monkeypatch, reply)

    assert launch._variant_for_other_gpu() == expected


@pytest.mark.parametrize("reply, expected", [("1", "cuda13.0"), ("", "cpu"), ("3", "cpu")])
def test_undetected_gpu_should_let_the_user_choose(launch, monkeypatch, reply, expected):
    answer(monkeypatch, reply)

    assert launch._variant_for_undetected_gpu(None) == (expected, False, None)


def test_amd_should_use_rocm_for_a_supported_card(launch, monkeypatch):
    monkeypatch.setattr(launch, "detect_amd_gfx_version", lambda name: ("gfx110X-dgpu", "RDNA 3", True))
    answer(monkeypatch, "1")

    assert launch._variant_for_amd("AMD Radeon RX 7900 XTX") == ("rocm7.2.1", True, "gfx110X-dgpu")


def test_amd_should_fall_back_to_cpu_for_an_unsupported_card(launch, monkeypatch):
    monkeypatch.setattr(launch, "detect_amd_gfx_version", lambda name: ("gfx103X-dgpu", "RDNA 2", False))
    monkeypatch.setattr(launch, "choose_when_amd_unsupported", lambda: "cpu")

    assert launch._variant_for_amd("AMD Radeon RX 6800") == ("cpu", False, None)


def test_explicit_request_should_reject_an_unknown_variant(launch):
    assert launch._variant_for_explicit_request(SimpleNamespace(requirements="no-such-variant"), None) is None


def test_explicit_request_should_keep_a_known_variant(launch):
    assert launch._variant_for_explicit_request(SimpleNamespace(requirements="cpu"), None) == ("cpu", False, None)
