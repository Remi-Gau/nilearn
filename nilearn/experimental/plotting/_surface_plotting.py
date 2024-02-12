"""Experimental Surface plotting."""

from __future__ import annotations

import numpy

from nilearn import plotting as old_plotting
from nilearn._utils import fill_doc
from nilearn.experimental.surface import Mesh, PolyMesh, SurfaceImage


@fill_doc
def plot_surf(
    surf_map: SurfaceImage | str | numpy.array | None = None,
    hemi: str = "left",
    surf_mesh: (
        str | list[numpy.array, numpy.array] | Mesh | PolyMesh | None
    ) = None,
    bg_map: str | numpy.array | SurfaceImage | None = None,
    **kwargs,
):
    """Plot a SurfaceImage.

    Parameters
    ----------
    surf_map : SurfaceImage or :obj:`str` or numpy.ndarray, optional

    surf_mesh : :obj:`str` or :obj:`list` of two numpy.ndarray \
                or Mesh or PolyMesh, optional

    %(hemi)s
    """
    if isinstance(bg_map, SurfaceImage):
        bg_map._check_hemi(hemi)
        bg_map = bg_map.data[hemi]

    if not isinstance(surf_map, SurfaceImage):
        return old_plotting.plot_surf(
            surf_mesh=surf_mesh,
            surf_map=surf_map,
            hemi=hemi,
            bg_map=bg_map**kwargs,
        )

    if surf_mesh is None:
        surf_mesh = surf_map.mesh

    surf_map._check_hemi(hemi)
    _check_hemi_present(surf_mesh, hemi)

    return old_plotting.plot_surf(
        surf_mesh=surf_mesh[hemi],
        surf_map=surf_map.data[hemi],
        hemi=hemi,
        bg_map=bg_map,
        **kwargs,
    )


@fill_doc
def plot_surf_stat_map(
    stat_map: SurfaceImage | str | numpy.array | None = None,
    hemi: str = "left",
    surf_mesh: (
        str | list[numpy.array, numpy.array] | Mesh | PolyMesh | None
    ) = None,
    bg_map: str | numpy.array | SurfaceImage | None = None,
    **kwargs,
):
    """Plot a stats map on a SurfaceImage.

    Parameters
    ----------
    stat_map: SurfaceImage or :obj:`str` or numpy.ndarray, optional

    surf_mesh : :obj:`str` or :obj:`list` of two numpy.ndarray \
                or Mesh or PolyMesh, optional

    %(hemi)s
    """
    if isinstance(bg_map, SurfaceImage):
        bg_map._check_hemi(hemi)
        bg_map = bg_map.data[hemi]

    if not isinstance(stat_map, SurfaceImage):
        return old_plotting.plot_surf_stat_map(
            surf_mesh=surf_mesh,
            stat_map=stat_map,
            hemi=hemi,
            bg_map=bg_map,
            **kwargs,
        )

    if surf_mesh is None:
        surf_mesh = stat_map.mesh

    stat_map._check_hemi(hemi)
    _check_hemi_present(surf_mesh, hemi)

    fig = old_plotting.plot_surf_stat_map(
        surf_mesh=surf_mesh[hemi],
        stat_map=stat_map.data[hemi],
        hemi=hemi,
        bg_map=bg_map,
        **kwargs,
    )
    return fig


@fill_doc
def plot_surf_contours(
    roi_map: SurfaceImage | str | numpy.array | list[numpy.ndarray],
    hemi: str = "left",
    surf_mesh: (
        str | list[numpy.array, numpy.array] | Mesh | PolyMesh | None
    ) = None,
    **kwargs,
):
    """Plot a contours on a SurfaceImage.

    Parameters
    ----------
    roi_map : SurfaceImage or :obj:`str` or numpy.ndarray \
              or :obj:`list` of numpy.ndarray

    surf_mesh : :obj:`str` or :obj:`list` of two numpy.ndarray \
                or Mesh or PolyMesh, optional

    %(hemi)s
    """
    if not isinstance(roi_map, SurfaceImage):
        return old_plotting.plot_surf_contours(
            surf_mesh=surf_mesh,
            roi_map=roi_map,
            hemi=hemi,
            **kwargs,
        )

    if surf_mesh is None:
        surf_mesh = roi_map.mesh

    roi_map._check_hemi(hemi)
    _check_hemi_present(surf_mesh, hemi)

    fig = old_plotting.plot_surf_contours(
        surf_mesh=surf_mesh[hemi],
        roi_map=roi_map.data[hemi],
        hemi=hemi,
        **kwargs,
    )
    return fig


@fill_doc
def plot_surf_roi(
    roi_map: SurfaceImage | str | numpy.array | list[numpy.ndarray],
    hemi: str = "left",
    surf_mesh: (
        str | list[numpy.array, numpy.array] | Mesh | PolyMesh | None
    ) = None,
    bg_map: str | numpy.array | SurfaceImage | None = None,
    **kwargs,
):
    """Plot a contours on a SurfaceImage.

    Parameters
    ----------
    roi_map : SurfaceImage or :obj:`str` or numpy.ndarray \
              or :obj:`list` of numpy.ndarray

    surf_mesh : :obj:`str` or :obj:`list` of two numpy.ndarray \
                or Mesh or PolyMesh, optional

    %(hemi)s
    """
    if isinstance(bg_map, SurfaceImage):
        bg_map._check_hemi(hemi)
        bg_map = bg_map.data[hemi]

    if not isinstance(roi_map, SurfaceImage):
        return old_plotting.plot_surf_roi(
            surf_mesh=surf_mesh,
            roi_map=roi_map,
            hemi=hemi,
            bg_map=bg_map,
            **kwargs,
        )

    if surf_mesh is None:
        surf_mesh = roi_map.mesh

    roi_map._check_hemi(hemi)
    _check_hemi_present(surf_mesh, hemi)

    fig = old_plotting.plot_surf_roi(
        surf_mesh=surf_mesh[hemi],
        roi_map=roi_map.data[hemi],
        hemi=hemi,
        bg_map=bg_map,
        **kwargs,
    )
    return fig


@fill_doc
def view_surf(
    surf_map: SurfaceImage | str | numpy.array | None = None,
    hemi: str = "left",
    surf_mesh: (
        str | list[numpy.array, numpy.array] | Mesh | PolyMesh | None
    ) = None,
    bg_map: str | numpy.array | SurfaceImage | None = None,
    **kwargs,
):
    """View SurfaceImage in browser.

    Parameters
    ----------
    surf_map : SurfaceImage or :obj:`str` or numpy.ndarray, optional

    surf_mesh : :obj:`str` or :obj:`list` of two numpy.ndarray \
                or Mesh or PolyMesh, optional

    %(hemi)s
    """
    if isinstance(bg_map, SurfaceImage):
        bg_map._check_hemi(hemi)
        bg_map = bg_map.data[hemi]

    if not isinstance(surf_map, SurfaceImage):
        return old_plotting.view_surf(
            surf_mesh=surf_mesh,
            surf_map=surf_map,
            bg_map=bg_map,
            **kwargs,
        )

    if surf_mesh is None:
        surf_mesh = surf_map.mesh

    surf_map._check_hemi(hemi)
    _check_hemi_present(surf_mesh, hemi)

    fig = old_plotting.view_surf(
        surf_mesh=surf_mesh[hemi],
        surf_map=surf_map.data[hemi],
        bg_map=bg_map,
        **kwargs,
    )
    return fig


def _check_hemi_present(mesh: PolyMesh, hemi: str):
    """Check that a given hemisphere exists both in data and mesh."""
    if hemi not in mesh:
        raise ValueError(f"{hemi} must be present in mesh")
