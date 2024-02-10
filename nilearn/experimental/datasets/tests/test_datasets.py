import pytest

from nilearn.experimental.datasets import (
    fetch_destrieux,
    fetch_nki,
    load_fsaverage,
)


def test_load_fsaverage():
    """Call default function smoke test and assert return."""
    meshes = load_fsaverage()
    assert isinstance(meshes, dict)
    assert meshes["pial"]["left"].n_vertices == 10242  # fsaverage5


def test_load_fsaverage_wrong_mesh_name():
    """Give incorrect value to mesh_name argument."""
    with pytest.raises(ValueError, match="'mesh' should be one of"):
        load_fsaverage(mesh_name="foo")


def test_load_fsaverage_hemispheres_have_file():
    """Make sure file paths are present."""
    meshes = load_fsaverage()
    left_hemisphere_meshes = [
        mesh for mesh in meshes.values() if "left" in mesh
    ]
    assert left_hemisphere_meshes
    right_hemisphere_meshes = [
        mesh for mesh in meshes.values() if "right" in mesh
    ]
    assert right_hemisphere_meshes


def test_destrieux_nki_wrong_mesh_type():
    with pytest.raises(ValueError, match="'mesh_type' must be one of"):
        fetch_nki(mesh_type="foo")

    with pytest.raises(ValueError, match="'mesh_type' must be one of"):
        fetch_destrieux(mesh_type="foo")
