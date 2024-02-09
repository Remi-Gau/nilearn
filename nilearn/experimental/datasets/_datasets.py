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
    meshes: dict[str, dict[str, Mesh]] = {}
    renaming = {
        "pial": "pial",
        "white": "white_matter",
        "infl": "inflated",
        "sphere": "sphere",
        "flat": "flat",
    }
    for mesh_type, mesh_name in renaming.items():
        meshes[mesh_name] = {}
        for hemisphere in "left", "right":
            meshes[mesh_name][f"{hemisphere}"] = FileMesh(
                fsaverage[f"{mesh_type}_{hemisphere}"]
            )
    data = {}
    renaming = {"curv": "curvature", "sulc": "sulcal", "thick": "thickness"}
    for data_type, data_name in renaming.items():
        data[data_name] = {}
        for hemisphere in "left", "right":
            data[data_name][f"{hemisphere}"] = fsaverage[
                f"{data_type}_{hemisphere}"
            ]

    return meshes, data


def fetch_nki(n_subjects=1) -> Sequence[SurfaceImage]:
    """Load NKI enhanced surface data into a surface object."""
    fsaverage, _ = load_fsaverage("fsaverage5")
    nki_dataset = datasets.fetch_surf_nki_enhanced(n_subjects=n_subjects)
    images = []
    for left, right in zip(
        nki_dataset["func_left"], nki_dataset["func_right"]
    ):
        left_data = _io.read_array(left).T
        right_data = _io.read_array(right).T
        img = SurfaceImage(
            mesh=fsaverage["pial"],
            data={
                "left": left_data,
                "right": right_data,
            },
        )
        images.append(img)
    return images


def fetch_destrieux() -> tuple[SurfaceImage, dict[int, str]]:
    """Load Destrieux surface atlas into a surface object."""
    fsaverage, _ = load_fsaverage("fsaverage5")
    destrieux = datasets.fetch_atlas_surf_destrieux()
    # TODO fetchers usually return Bunch
    return (
        SurfaceImage(
            mesh=fsaverage["pial"],
            data={
                "left": destrieux["map_left"],
                "right": destrieux["map_right"],
            },
        ),
        destrieux.labels,
    )
