import _bootstrap  # noqa: F401, I001

import json

import numpy as np

from editor.paste_overlay_state import (
    PAGE_KEY,
    compose_paste_overlays,
    new_overlay_id,
    normalize_paste_overlay,
    parse_page_paste_overlays,
    png_base64_to_rgba_overlay,
    rgba_overlay_to_png_base64,
    serialize_paste_overlays,
)


def _valid_overlay(**overrides):
    data = {
        "id": "ovl-1",
        "name": "特效字",
        "z": 2,
        "visible": True,
        "opacity": 0.6,
        "center_x": 12.5,
        "center_y": -3.25,
        "width": 64.0,
        "height": 32.0,
        "rotation": 15.0,
        "flip_h": True,
        "flip_v": False,
        "image": "",
    }
    data.update(overrides)
    return data


def test_normalize_fills_defaults_and_coerces():
    raw = {"width": "80", "height": "40", "z": "1", "visible": 1, "opacity": "0.5"}
    overlay = normalize_paste_overlay(raw)
    assert overlay["width"] == 80.0
    assert overlay["height"] == 40.0
    assert overlay["z"] == 1
    assert overlay["visible"] is True
    assert overlay["opacity"] == 0.5
    assert overlay["name"] == "贴片"
    assert overlay["image"] == ""
    assert overlay["center_x"] == 0.0
    assert overlay["center_y"] == 0.0
    assert overlay["rotation"] == 0.0
    assert overlay["flip_h"] is False
    assert overlay["flip_v"] is False
    assert overlay["id"]


def test_opacity_is_clamped_to_unit_range():
    assert normalize_paste_overlay(_valid_overlay(opacity=-1))["opacity"] == 0.0
    assert normalize_paste_overlay(_valid_overlay(opacity=3.5))["opacity"] == 1.0


def test_geometry_must_be_positive():
    for kwargs in ({"width": 0}, {"height": -1}, {"width": -5.0, "height": 3.0}):
        try:
            normalize_paste_overlay(_valid_overlay(**kwargs))
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def test_invalid_image_payload_raises():
    try:
        normalize_paste_overlay(_valid_overlay(image="not-base64!!"))
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid base64 image")


def test_boolean_strings_parsed_explicitly():
    assert normalize_paste_overlay(_valid_overlay(visible="false"))["visible"] is False
    assert normalize_paste_overlay(_valid_overlay(visible="true"))["visible"] is True
    try:
        normalize_paste_overlay(_valid_overlay(visible="sometimes"))
    except ValueError:
        return
    raise AssertionError("expected ValueError for unparsable boolean string")


def test_compose_skips_decompression_bomb_png_before_decode():
    import base64 as _base64
    import struct as _struct
    import zlib as _zlib

    # Build a small PNG, then rewrite the IHDR width and height to huge values (recomputing the CRC so it stays valid),
    # to check that compositing skips it before imdecode
    tiny = rgba_overlay_to_png_base64(_solid_rgba(2, 2, (255, 0, 0)))
    raw = bytearray(_base64.b64decode(tiny))
    raw[16:24] = _struct.pack(">II", 200_000, 200_000)
    raw[29:33] = _struct.pack(">I", _zlib.crc32(raw[12:29]) & 0xFFFFFFFF)
    overlay = _valid_overlay(image=_base64.b64encode(bytes(raw)).decode("ascii"))
    assert compose_paste_overlays([overlay], (80, 60)) is None


def test_serialize_assigns_unique_ids():
    overlays = serialize_paste_overlays(
        [
            _valid_overlay(id="same"),
            _valid_overlay(id="same"),
            _valid_overlay(id=""),
        ]
    )
    ids = [item["id"] for item in overlays]
    assert len(ids) == len(set(ids))
    assert ids[0] == "same"


def test_parse_missing_key_returns_empty():
    assert parse_page_paste_overlays({"regions": []}) == []
    assert parse_page_paste_overlays({"paste_overlays": []}) == []


def test_parse_skips_invalid_entries():
    page = {
        "paste_overlays": [
            _valid_overlay(id="keep-1"),
            {"width": -1, "height": 5},  # Invalid geometry -> skipped
            _valid_overlay(id="keep-2"),
        ]
    }
    parsed = parse_page_paste_overlays(page)
    assert [item["id"] for item in parsed] == ["keep-1", "keep-2"]


def test_json_round_trip_keeps_structure():
    overlays = serialize_paste_overlays(
        [_valid_overlay(), _valid_overlay(id="", name="背景补块", z=0)]
    )
    text = json.dumps(
        {"regions": [], PAGE_KEY: overlays}, ensure_ascii=False
    )
    loaded_page = json.loads(text)
    parsed = parse_page_paste_overlays(loaded_page)
    assert parsed == overlays


def test_png_base64_helpers_round_trip():
    rgba = np.zeros((6, 5, 4), dtype=np.uint8)
    rgba[..., 0] = np.arange(6 * 5).reshape(6, 5) % 256
    rgba[..., 1] = 128
    rgba[..., 2] = 64
    rgba[..., 3] = 200
    encoded = rgba_overlay_to_png_base64(rgba)
    assert isinstance(encoded, str) and encoded
    decoded = png_base64_to_rgba_overlay(encoded)
    assert decoded is not None
    assert decoded.shape == rgba.shape
    assert decoded.dtype == np.uint8
    assert np.array_equal(decoded, rgba)
    assert not decoded.flags.writeable
    assert png_base64_to_rgba_overlay("") is None
    assert png_base64_to_rgba_overlay("not-base64!!") is None


def test_ids_survive_round_trip_without_duplicates():
    first = new_overlay_id()
    overlays = serialize_paste_overlays([_valid_overlay(id=first)])
    parsed = parse_page_paste_overlays({"paste_overlays": overlays})
    assert parsed[0]["id"] == first


def _solid_rgba(width, height, color):
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
    rgba[..., 3] = 255
    return rgba


def _overlay_with_image(center_x, center_y, width=10, height=10, **overrides):
    image = rgba_overlay_to_png_base64(_solid_rgba(width, height, (255, 0, 0)))
    data = {
        "image": image,
        "center_x": center_x,
        "center_y": center_y,
        "width": width,
        "height": height,
        "rotation": 0.0,
        "flip_h": False,
        "flip_v": False,
        "opacity": 1.0,
    }
    data.update(overrides)
    return _valid_overlay(**data)


def test_compose_paste_overlays_places_image_at_center():
    composite = compose_paste_overlays([_overlay_with_image(20, 30)], (80, 60))
    assert composite is not None
    assert composite.shape == (60, 80, 4)
    # The centre should be opaque red
    assert composite[30, 20, 0] > 200
    assert composite[30, 20, 1] < 50
    assert composite[30, 20, 3] > 200
    # Positions far from the overlay stay transparent
    assert composite[5, 5, 3] == 0


def test_compose_paste_overlays_honors_visibility_and_opacity():
    invisible = _overlay_with_image(20, 30, visible=False)
    assert compose_paste_overlays([invisible], (80, 60)) is None

    faded = _overlay_with_image(20, 30, opacity=0.5)
    composite = compose_paste_overlays([faded], (80, 60))
    assert composite is not None
    assert 60 < composite[30, 20, 3] < 200


def test_compose_paste_overlays_rotation_smoke():
    rotated = _overlay_with_image(40, 30, width=20, height=10, rotation=45)
    composite = compose_paste_overlays([rotated], (80, 60))
    assert composite is not None
    assert np.any(composite[..., 3] > 0)


def _solid_overlay(color, center, size=10, z=0, **overrides):
    image = rgba_overlay_to_png_base64(_solid_rgba(size, size, color))
    data = {
        "visible": True,
        "image": image,
        "center_x": center[0],
        "center_y": center[1],
        "width": size,
        "height": size,
        "rotation": 0.0,
        "flip_h": False,
        "flip_v": False,
        "opacity": 1.0,
        "z": z,
    }
    data.update(overrides)
    return _valid_overlay(**data)


def test_compose_respects_persisted_z_order():
    # The list order is the reverse of z on purpose: the red with z=1 should cover the green with z=0
    green_low = _solid_overlay((0, 255, 0), (30, 30), z=0)
    red_high = _solid_overlay((255, 0, 0), (30, 30), z=1)
    composite = compose_paste_overlays([red_high, green_low], (80, 60))
    assert composite is not None
    assert composite[30, 30, 0] > 200  # The top layer is red
    assert composite[30, 30, 1] < 60


def test_compose_source_over_blends_semi_transparent_overlay():
    red_bottom = _solid_overlay((255, 0, 0), (30, 30), z=0)
    blue_half = _solid_overlay((0, 0, 255), (30, 30), z=1, opacity=0.5)
    composite = compose_paste_overlays([red_bottom, blue_half], (80, 60))
    assert composite is not None
    # source-over: 50% blue over opaque red -> magenta, opaque as a whole
    assert composite[30, 30, 3] > 250
    assert 100 < composite[30, 30, 0] < 160
    assert composite[30, 30, 1] < 40
    assert 100 < composite[30, 30, 2] < 160


def test_compose_premultiplies_before_affine_interpolation():
    # The source is 2x1: opaque red on the left, fully transparent on the right; enlarging 16 times horizontally produces semi-transparent transition pixels.
    # Premultiplying before interpolation keeps black from bleeding into the transparent edge (straight-alpha interpolation would darken the RGB with it).
    source = np.zeros((1, 2, 4), dtype=np.uint8)
    source[0, 0] = (255, 0, 0, 255)
    overlay = _valid_overlay(
        image=rgba_overlay_to_png_base64(source),
        center_x=20.0,
        center_y=0.5,
        width=32.0,
        height=1.0,
        rotation=0.0,
        flip_h=False,
        flip_v=False,
        opacity=1.0,
    )
    composite = compose_paste_overlays([overlay], (40, 2))
    assert composite is not None
    partial = np.where((composite[..., 3] > 0) & (composite[..., 3] < 255))
    assert len(partial[0]) > 0, "应存在半透明过渡像素"
    reds = composite[partial[0], partial[1], 0]
    assert np.all(reds > 200), f"半透明边缘反预乘后不应变暗: {reds}"


def test_compose_large_canvas_small_overlay_uses_bounded_box():
    # Large canvas + small overlay: check that the bounding-box path gives the right result and does not crash out of bounds (memory is cut to the bounding box)
    overlay = _overlay_with_image(1024, 1024, width=16, height=16)
    composite = compose_paste_overlays([overlay], (2048, 2048))
    assert composite is not None
    assert composite.shape == (2048, 2048, 4)
    assert composite[1024, 1024, 3] > 250
    assert composite[1024, 1024, 0] > 200


def test_compose_clamps_overlay_partially_outside_canvas():
    # The overlay centre is outside the canvas but part of it is visible: the bounding box is clipped to the canvas, with no crash and the visible pixels kept
    overlay = _overlay_with_image(-4, -4, width=16, height=16)
    composite = compose_paste_overlays([overlay], (40, 30))
    assert composite is not None
    assert composite[0, 0, 3] > 250
    assert composite[0, 0, 0] > 200


def test_paste_overlay_supported_suffixes_aligns_with_global():
    from desktop_qt_ui.ui.editor.graphics_view_paste_overlays import _IMAGE_SUFFIXES
    from manga_translator.image_formats import SUPPORTED_IMAGE_EXTENSIONS

    assert set(SUPPORTED_IMAGE_EXTENSIONS).issubset(_IMAGE_SUFFIXES)
    assert ".jfif" in _IMAGE_SUFFIXES
    assert ".png" in _IMAGE_SUFFIXES
    assert ".webp" in _IMAGE_SUFFIXES
    assert ".bmp" in _IMAGE_SUFFIXES
