from nilearn import plotting as old_plotting
from nilearn.experimental.surface import PolyMesh, SurfaceImage


def plot_surf(
    img: SurfaceImage,
    hemi: str | None = None,
    surf_mesh: PolyMesh | None = None,
    **kwargs,
):
    """Plot a SurfaceImage.

    TODO: docstring.
    """
    # TODO deal with surf_mesh, surf_map, hemi in kwargs
    if surf_mesh is None:
        surf_mesh = img.mesh

    if hemi is None:
        hemi = "left"

    _check_hemi_present(surf_mesh, img, hemi)

    fig = old_plotting.plot_surf(
        surf_mesh=surf_mesh[hemi],
        surf_map=img.data[hemi],
        hemi=hemi,
        **kwargs,
    )
    return fig


def plot_surf_stat_map(
    img: SurfaceImage,
    hemi: str | None = None,
    surf_mesh: PolyMesh | None = None,
    **kwargs,
):
    """Plot a stats map on a SurfaceImage.

    TODO: docstring.
    """
    # TODO deal with stat_map in kwargs
    if surf_mesh is None:
        surf_mesh = img.mesh

    if hemi is None:
        hemi = "left"

    _check_hemi_present(surf_mesh, img, hemi)

    fig = old_plotting.plot_surf_stat_map(
        surf_mesh=surf_mesh[hemi],
        stat_map=img.data[hemi],
        hemi=hemi,
        **kwargs,
    )
    return fig


def plot_surf_contours(
    img: SurfaceImage,
    hemi: str | None = None,
    surf_mesh: PolyMesh | None = None,
    **kwargs,
):
    """Plot a contours on a SurfaceImage.

    TODO: docstring.
    """
    # TODO deal with roi_map in kwargs
    if surf_mesh is None:
        surf_mesh = img.mesh

    if hemi is None:
        hemi = "left"

    _check_hemi_present(surf_mesh, img, hemi)

    fig = old_plotting.plot_surf_contours(
        surf_mesh=surf_mesh[hemi],
        roi_map=img.data[hemi],
        hemi=hemi,
        **kwargs,
    )
    return fig


def view_surf(
    img: SurfaceImage,
    hemi: str | None = None,
    surf_mesh: PolyMesh | None = None,
    **kwargs,
):
    """View SurfaceImage in browser.

    TODO: docstring.
    """
    if surf_mesh is None:
        surf_mesh = img.mesh

    if hemi is None:
        hemi = "left"

    _check_hemi_present(surf_mesh, img, hemi)

    fig = old_plotting.view_surf(
        surf_mesh=surf_mesh[hemi], surf_map=img.data[hemi], **kwargs
    )
    return fig


def _check_hemi_present(mesh: PolyMesh, img: SurfaceImage, hemi: str):
    if hemi not in mesh or hemi not in img.data:
        raise ValueError(
            f"{hemi} must be present in mesh and SurfaceImage data"
        )
