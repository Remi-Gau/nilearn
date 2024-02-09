from typing import Callable

import numpy as np
import pytest

from nilearn.experimental.datasets import load_fsaverage
from nilearn.experimental.surface import InMemoryMesh, PolyMesh, SurfaceImage


@pytest.fixture
def mini_mesh() -> PolyMesh:
    """Small mesh for tests with 2 parts with different numbers of vertices."""
    left_coords = np.asarray([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    left_faces = np.asarray([[1, 0, 2], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    right_coords = (
        np.asarray([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
        + 2.0
    )
    right_faces = np.asarray(
        [
            [0, 1, 4],
            [0, 3, 1],
            [1, 3, 2],
            [1, 2, 4],
            [2, 3, 4],
            [0, 4, 3],
        ]
    )
    return {
        "left": InMemoryMesh(left_coords, left_faces),
        "right": InMemoryMesh(right_coords, right_faces),
    }


@pytest.fixture
def make_mini_surface_img(mini_mesh) -> Callable:
    """Small surface image for tests"""

    def f(shape=()):
        data = {}
        for i, (key, val) in enumerate(mini_mesh.items()):
            data_shape = tuple(shape) + (val.n_vertices,)
            data_part = (
                np.arange(np.prod(data_shape)).reshape(data_shape) + 1.0
            ) * 10**i
            data[key] = data_part
        return SurfaceImage(mini_mesh, data)

    return f


@pytest.fixture
def mini_surface_mask(mini_surface_img) -> SurfaceImage:
    data = {k: (v > v.ravel()[0]) for k, v in mini_surface_img.data.items()}
    return SurfaceImage(mini_surface_img.mesh, data)


@pytest.fixture
def mini_surface_img(make_mini_surface_img) -> SurfaceImage:
    return make_mini_surface_img()


@pytest.fixture
def flip():
    def f(parts):
        if not parts:
            return {}
        keys = list(parts.keys())
        keys = [keys[-1]] + keys[:-1]
        return dict(zip(keys, parts.values()))

    return f


@pytest.fixture
def flip_img(flip):
    def f(img):
        return SurfaceImage(flip(img.mesh), flip(img.data))

    return f


@pytest.fixture
def pial_surface_mesh():
    """Get fsaverage mesh for testing."""
    return load_fsaverage()["pial"]


@pytest.fixture
def assert_img_equal():
    def f(img_1, img_2):
        assert set(img_1.data.keys()) == set(img_2.data.keys())
        for key in img_1.data:
            assert np.array_equal(img_1.data[key], img_2.data[key])

    return f


@pytest.fixture
def drop_img_part():
    def f(img, part_name="right"):
        mesh = img.mesh.copy()
        mesh.pop(part_name)
        data = img.data.copy()
        data.pop(part_name)
        return SurfaceImage(mesh, data)

    return f
