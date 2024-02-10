"""Fetching a few example datasets to use during development.

eventually nilearn.datasets would be updated
"""

from __future__ import annotations

from typing import Sequence

from nilearn import datasets
from nilearn.experimental.surface import _io
from nilearn.experimental.surface._surface_image import (
    FileMesh,
    Mesh,
    PolyMesh,
    SurfaceImage,
)


def load_fsaverage(
    mesh_name: str = "fsaverage5",
) -> tuple[dict[str, PolyMesh], dict[str, str]]:
    """Load fsaverage for both hemispheres."""
    fsaverage = datasets.fetch_surf_fsaverage(mesh_name)
    renaming = {
        "pial": "pial",
        "white": "white_matter",
        "infl": "inflated",
        "sphere": "sphere",
        "flat": "flat",
    }
    meshes: dict[str, dict[str, Mesh]] = {
        mesh_name: {
            f"{hemisphere}": FileMesh(fsaverage[f"{mesh_type}_{hemisphere}"])
            for hemisphere in ("left", "right")
        }
        for mesh_type, mesh_name in renaming.items()
    }
    renaming = {"curv": "curvature", "sulc": "sulcal", "thick": "thickness"}
    data = {
        data_name: {
            f"{hemisphere}": fsaverage[f"{data_type}_{hemisphere}"]
            for hemisphere in ("left", "right")
        }
        for data_type, data_name in renaming.items()
    }
    return meshes, data


ALLOWED_MESH_TYPES = (
    "pial",
    "white_matter",
    "inflated",
    "sphere",
    "flat",
)


def fetch_nki(mesh_type: str = "pial", **kwargs) -> Sequence[SurfaceImage]:
    """Load NKI enhanced surface data into a surface object."""
    if mesh_type not in ALLOWED_MESH_TYPES:
        raise ValueError(
            f"'mesh_type' must be one of {ALLOWED_MESH_TYPES}.\n"
            f"Got: {mesh_type}."
        )

    fsaverage, _ = load_fsaverage("fsaverage5")

    nki_dataset = datasets.fetch_surf_nki_enhanced(**kwargs)

    images = []
    for left, right in zip(
        nki_dataset["func_left"], nki_dataset["func_right"]
    ):
        left_data = _io.read_array(left).T
        right_data = _io.read_array(right).T
        img = SurfaceImage(
            mesh=fsaverage[mesh_type],
            data={
                "left": left_data,
                "right": right_data,
            },
        )
        images.append(img)

    return images


def fetch_destrieux(
    mesh_type: str = "pial",
) -> tuple[SurfaceImage, dict[int, str]]:
    """Load Destrieux surface atlas into a surface object."""
    if mesh_type not in ALLOWED_MESH_TYPES:
        raise ValueError(
            f"'mesh_type' must be one of {ALLOWED_MESH_TYPES}.\n"
            f"Got: {mesh_type}."
        )

    fsaverage, _ = load_fsaverage("fsaverage5")
    destrieux = datasets.fetch_atlas_surf_destrieux()
    # TODO fetchers usually return Bunch
    return (
        SurfaceImage(
            mesh=fsaverage[mesh_type],
            data={
                "left": destrieux["map_left"],
                "right": destrieux["map_right"],
            },
        ),
        destrieux.labels,
    )
