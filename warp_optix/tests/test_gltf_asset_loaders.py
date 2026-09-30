import numpy as np
import pytest

from warp_optix.pathtracing.asset_loaders import _build_texture_list, _load_gltf_images


def test_missing_gltf_image_does_not_shift_texture_sources():
    image = np.zeros((2, 2, 4), dtype=np.float32)
    image[..., 0] = 0.25
    textures = _build_texture_list(
        {"textures": [{"source": 0}, {"source": 1}]}, [None, image]
    )

    np.testing.assert_array_equal(textures[0], np.ones((1, 1, 4), dtype=np.float32))
    assert textures[1] is image


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_gltf_image_normalization_uses_source_bit_depth(monkeypatch, tmp_path, dtype):
    imageio = pytest.importorskip("imageio.v3")
    maximum = np.iinfo(dtype).max
    monkeypatch.setattr(
        imageio,
        "imread",
        lambda _stream: np.full((1, 1, 3), maximum, dtype=dtype),
    )
    images = _load_gltf_images(
        tmp_path,
        {"images": [{"uri": "data:image/png;base64,AA=="}]},
        [],
    )
    np.testing.assert_array_equal(images[0], np.ones((1, 1, 4), dtype=np.float32))
