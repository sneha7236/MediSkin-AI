import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import database  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    """Fresh SQLite database per test."""
    database.init_engine(f"sqlite:///{tmp_path / 'test.db'}")
    yield
    database.get_engine().dispose()


def make_image_bytes(fmt="PNG", size=(64, 64), color=(180, 120, 90)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture()
def png_bytes():
    return make_image_bytes("PNG")


@pytest.fixture()
def jpg_bytes():
    return make_image_bytes("JPEG")
