"""
Server utility functions
"""
import io
import secrets

from fastapi import HTTPException

from manga_translator.server.to_json import to_translation


def generate_nonce():
    """Generate a random nonce"""
    return secrets.token_hex(16)


def transform_to_image(ctx):
    """Convert a translation context to image bytes"""
    # Check whether ctx.result exists
    if ctx.result is None:
        raise HTTPException(500, detail="Translation failed: no result image generated")
    
    # Check whether a placeholder is used (in web mode this flag is set after final.png is saved)
    if hasattr(ctx, 'use_placeholder') and ctx.use_placeholder:
        # ctx.result is already a 1x1 placeholder image, which transfers quickly
        img_byte_arr = io.BytesIO()
        ctx.result.save(img_byte_arr, format="PNG")
        return img_byte_arr.getvalue()

    # Return the full translation result
    img_byte_arr = io.BytesIO()
    ctx.result.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


def transform_to_json(ctx):
    """Convert a translation context to JSON bytes"""
    return to_translation(ctx).model_dump_json().encode("utf-8")


def transform_to_bytes(ctx):
    """Convert a translation context to bytes"""
    return to_translation(ctx).to_bytes()
