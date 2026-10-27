"""Input validation for forms and uploaded images."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from config import constants
from config import config

Image.MAX_IMAGE_PIXELS = constants.MAX_IMAGE_PIXELS

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}$")


class ValidationError(ValueError):
    """Raised with a user-friendly message when input is rejected."""


def validate_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not email:
        raise ValidationError("Email is required.")
    if len(email) > constants.MAX_EMAIL_LENGTH or not _EMAIL_RE.match(email):
        raise ValidationError("Please enter a valid email address.")
    return email


def validate_name(name: str) -> str:
    name = re.sub(r"\s+", " ", (name or "").strip())
    if not name:
        raise ValidationError("Full name is required.")
    if len(name) > constants.MAX_NAME_LENGTH:
        raise ValidationError(f"Name must be at most {constants.MAX_NAME_LENGTH} characters.")
    if re.search(r"[<>]", name):
        raise ValidationError("Name contains invalid characters.")
    return name


def validate_password_strength(password: str) -> None:
    if not password:
        raise ValidationError("Password is required.")
    problems = []
    if len(password) < constants.MIN_PASSWORD_LENGTH:
        problems.append(f"at least {constants.MIN_PASSWORD_LENGTH} characters")
    if not re.search(r"[a-z]", password):
        problems.append("a lowercase letter")
    if not re.search(r"[A-Z]", password):
        problems.append("an uppercase letter")
    if not re.search(r"\d", password):
        problems.append("a digit")
    if not re.search(r"[^A-Za-z0-9]", password):
        problems.append("a special character")
    if problems:
        raise ValidationError("Password must contain " + ", ".join(problems) + ".")


@dataclass(frozen=True)
class ValidatedImage:
    image: Image.Image  # decoded, RGB, fully loaded
    format: str
    size_bytes: int


def validate_image_upload(filename: str, data: bytes | None) -> ValidatedImage:
    """Validate extension, size, emptiness and decodability of an uploaded image.

    Returns a fully decoded RGB image. Raises ValidationError with a friendly message.
    """
    if not data:
        raise ValidationError("The uploaded file is empty.")
    extension = (filename or "").rsplit(".", 1)[-1].lower() if "." in (filename or "") else ""
    if extension not in constants.ALLOWED_EXTENSIONS:
        allowed = ", ".join(e.upper() for e in constants.ALLOWED_EXTENSIONS)
        raise ValidationError(f"Unsupported file type. Please upload one of: {allowed}.")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise ValidationError(f"File is too large. Maximum size is {config.MAX_UPLOAD_MB} MB.")

    try:
        with Image.open(io.BytesIO(data)) as probe:
            detected_format = probe.format
            probe.verify()
        if detected_format not in constants.ALLOWED_MIME_FORMATS:
            raise ValidationError("File content is not a valid JPEG or PNG image.")
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            rgb = img.convert("RGB")
    except ValidationError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise ValidationError("The image is corrupted or cannot be read.") from None

    width, height = rgb.size
    if min(width, height) < constants.MIN_IMAGE_SIDE:
        raise ValidationError(
            f"Image is too small. Minimum size is {constants.MIN_IMAGE_SIDE}x{constants.MIN_IMAGE_SIDE} pixels."
        )
    return ValidatedImage(image=rgb, format=detected_format, size_bytes=len(data))
