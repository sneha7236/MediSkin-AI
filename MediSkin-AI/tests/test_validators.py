import pytest

from config import config
from utils import security
from utils.validators import (
    ValidationError, validate_email, validate_image_upload, validate_name, validate_password_strength,
)


def test_password_hash_roundtrip_and_not_plaintext():
    hashed = security.hash_password("Str0ng!Pass")
    assert hashed != "Str0ng!Pass" and "Str0ng" not in hashed
    assert security.verify_password("Str0ng!Pass", hashed)
    assert not security.verify_password("wrong", hashed)
    assert not security.verify_password("x", "not-a-hash")


def test_hash_is_salted():
    assert security.hash_password("Str0ng!Pass") != security.hash_password("Str0ng!Pass")


@pytest.mark.parametrize("pw", ["", "Sh0rt!A", "alllowercase1!", "ALLUPPERCASE1!", "NoDigits!!aa", "NoSpecial123a"])
def test_weak_passwords_rejected(pw):
    with pytest.raises(ValidationError):
        validate_password_strength(pw)


def test_strong_password_ok():
    validate_password_strength("Str0ng!Pass")


def test_email_validation():
    assert validate_email("  Bob@Example.COM ") == "bob@example.com"
    for bad in ["", "nope", "a@b", "a@@b.com", "a b@c.com"]:
        with pytest.raises(ValidationError):
            validate_email(bad)


def test_name_validation():
    assert validate_name("  Jane   Doe ") == "Jane Doe"
    for bad in ["", "   ", "<script>"]:
        with pytest.raises(ValidationError):
            validate_name(bad)


def test_valid_images(png_bytes, jpg_bytes):
    assert validate_image_upload("a.png", png_bytes).image.mode == "RGB"
    assert validate_image_upload("a.JPG", jpg_bytes).format == "JPEG"
    assert validate_image_upload("a.jpeg", jpg_bytes).format == "JPEG"


def test_image_rejections(png_bytes, monkeypatch):
    with pytest.raises(ValidationError, match="empty"):
        validate_image_upload("a.png", b"")
    with pytest.raises(ValidationError, match="Unsupported"):
        validate_image_upload("a.gif", png_bytes)
    with pytest.raises(ValidationError, match="Unsupported"):
        validate_image_upload("noextension", png_bytes)
    with pytest.raises(ValidationError, match="corrupted"):
        validate_image_upload("a.png", b"not an image at all")
    with pytest.raises(ValidationError, match="corrupted"):
        validate_image_upload("a.png", png_bytes[: len(png_bytes) // 2])
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(ValidationError, match="too large"):
        validate_image_upload("a.png", png_bytes)


def test_content_type_mismatch_rejected():
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, format="BMP")
    with pytest.raises(ValidationError):
        validate_image_upload("fake.png", buf.getvalue())


def test_tiny_image_rejected():
    from tests.conftest import make_image_bytes
    with pytest.raises(ValidationError, match="too small"):
        validate_image_upload("a.png", make_image_bytes(size=(8, 8)))


def test_path_traversal_helper(tmp_path):
    assert security.is_within_directory(tmp_path / "a.pdf", tmp_path)
    assert not security.is_within_directory(tmp_path / ".." / "etc", tmp_path)
