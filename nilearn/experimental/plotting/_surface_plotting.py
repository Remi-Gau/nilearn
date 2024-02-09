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
    surf_mesh: str
    | list[numpy.array, numpy.array]
    | Mesh
    | PolyMesh
    | None = None,
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
    if not isinstance(surf_map, SurfaceImage):
        return old_plotting.plot_surf(
            surf_mesh=surf_mesh,
            surf_map=surf_map,
            hemi=hemi,
            **kwargs,
        )

    if surf_mesh is None:
        surf_mesh = surf_map.mesh

    _check_hemi_present(surf_mesh, surf_map, hemi)

    return old_plotting.plot_surf(
        surf_mesh=surf_mesh[hemi],
        surf_map=surf_map.data[hemi],
        hemi=hemi,
        **kwargs,
    )


@fill_doc
def plot_surf_stat_map(
    stat_map: SurfaceImage | str | numpy.array | None = None,
    hemi: str = "left",
    surf_mesh: str
    | list[numpy.array, numpy.array]
    | Mesh
    | PolyMesh
    | None = None,
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
    if not isinstance(stat_map, SurfaceImage):
        return old_plotting.plot_surf_stat_map(
            surf_mesh=surf_mesh,
            surf_map=stat_map,
            hemi=hemi,
            **kwargs,
        )

    if surf_mesh is None:
        surf_mesh = stat_map.mesh

    _check_hemi_present(surf_mesh, stat_map, hemi)

    fig = old_plotting.plot_surf_stat_map(
        surf_mesh=surf_mesh[hemi],
        stat_map=stat_map.data[hemi],
        hemi=hemi,
        **kwargs,
    )
    return fig


@fill_doc
def plot_surf_contours(
    roi_map: SurfaceImage | str | numpy.array | list[numpy.ndarray],
    hemi: str = "left",
    surf_mesh: str
    | list[numpy.array, numpy.array]
    | Mesh
    | PolyMesh
    | None = None,
    **kwargs,
):
    """Plot a contours on a SurfaceImage.

    Parameters
    ----------
    roi_map : SurfaceImage or str or numpy.ndarray or list of numpy.ndarray

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

    _check_hemi_present(surf_mesh, roi_map, hemi)

    fig = old_plotting.plot_surf_contours(
        surf_mesh=surf_mesh[hemi],
        roi_map=roi_map.data[hemi],
        hemi=hemi,
        **kwargs,
    )
    return fig


@fill_doc
def view_surf(
    surf_map: SurfaceImage | str | numpy.array | None = None,
    hemi: str = "left",
    surf_mesh: str
    | list[numpy.array, numpy.array]
    | Mesh
    | PolyMesh
    | None = None,
    **kwargs,
):
    """View SurfaceImage in browser.

    Parameters
    ----------
    surf_map : SurfaceImage or str or numpy.ndarray, optional

    surf_mesh : :obj:`str` or :obj:`list` of two numpy.ndarray \
                or Mesh or PolyMesh, optional

    %(hemi)s
    """
    if not isinstance(surf_map, SurfaceImage):
        return old_plotting.view_surf(
            surf_mesh=surf_mesh,
            surf_map=surf_map,
            **kwargs,
        )

    if surf_mesh is None:
        surf_mesh = surf_map.mesh

    _check_hemi_present(surf_mesh, surf_map, hemi)

    fig = old_plotting.view_surf(
        surf_mesh=surf_mesh[hemi], surf_map=surf_map.data[hemi], **kwargs
    )
    return fig


def _check_hemi_present(mesh: PolyMesh, img: SurfaceImage, hemi: str):
    """Check that a given hemisphere exists both in data and mesh."""
    if hemi not in mesh or hemi not in img.data:
        raise ValueError(
            f"{hemi} must be present in mesh and SurfaceImage data"
        )
