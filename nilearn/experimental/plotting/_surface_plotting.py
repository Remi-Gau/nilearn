import numpy as np
from matplotlib import pyplot as plt

from nilearn import plotting as old_plotting


def plot_surf(img, hemi=None, mesh=None, views=["lateral"], **kwargs):
    """Plot a SurfaceImage.

    TODO: docstring.
    """
    if mesh is None:
        mesh = img.mesh

    if hemi is None:
        hemi = list(img.data.keys())
    if isinstance(hemi, str):
        hemi = [hemi]

    fig, axes = plt.subplots(
        len(views),
        len(hemi),
        subplot_kw={"projection": "3d"},
        figsize=(4 * len(hemi), 4),
    )
    axes = np.atleast_2d(axes)

    title = ""
    if "title" in kwargs:
        title = f"{kwargs.pop('title')} - "

    for view, ax_row in zip(views, axes):
        for ax, mesh_part in zip(ax_row, hemi):
            # TODO
            # handle passing in kwargs:
            # - axes
            # - view
            old_plotting.plot_surf(
                mesh[mesh_part],
                img.data[mesh_part],
                hemi=mesh_part,
                view=view,
                axes=ax,
                title=title + mesh_part,
                **kwargs,
            )
    return fig


def plot_surf_stat_map(img, hemi=None, mesh=None, views=["lateral"], **kwargs):
    """Plot a stats map on a SurfaceImage.

    TODO: docstring.
    """
    if mesh is None:
        mesh = img.mesh

    if hemi is None:
        hemi = list(img.data.keys())
    if isinstance(hemi, str):
        hemi = [hemi]

    fig, axes = plt.subplots(
        len(views),
        len(hemi),
        subplot_kw={"projection": "3d"},
        figsize=(4 * len(hemi), 4),
    )
    axes = np.atleast_2d(axes)

    title = ""
    if "title" in kwargs:
        title = f"{kwargs.pop('title')} - "

    for view, ax_row in zip(views, axes):
        for ax, mesh_part in zip(ax_row, hemi):
            # TODO
            # handle passing in kwargs:
            # - axes
            # - view

            fig = old_plotting.plot_surf_stat_map(
                mesh[mesh_part],
                img.data[mesh_part],
                hemi=mesh_part,
                view=view,
                axes=ax,
                title=title + mesh_part,
                **kwargs,
            )
    return fig


def plot_surf_contours(img, hemi=None, mesh=None, views=["lateral"], **kwargs):
    """Plot a contours on a SurfaceImage.

    TODO: docstring.
    """
    if mesh is None:
        mesh = img.mesh

    if hemi is None:
        hemi = list(img.data.keys())
    if isinstance(hemi, str):
        hemi = [hemi]

    fig, axes = plt.subplots(
        len(views),
        len(hemi),
        subplot_kw={"projection": "3d"},
        figsize=(4 * len(hemi), 4),
    )
    axes = np.atleast_2d(axes)

    title = ""
    if "title" in kwargs:
        title = f"{kwargs.pop('title')} - "

    for view, ax_row in zip(views, axes):
        for ax, mesh_part in zip(ax_row, hemi):
            # TODO
            # handle passing in kwargs:
            # - axes
            # - view

            fig = old_plotting.plot_surf_contours(
                mesh[mesh_part],
                img.data[mesh_part],
                hemi=mesh_part,
                view=view,
                axes=ax,
                title=title + mesh_part,
                **kwargs,
            )
    return fig


def view_surf(img, surf_mesh, hemi, **kwargs):
    """View SurfaceImage in browser.

    TODO: docstring.
    """
    fig = old_plotting.view_surf(
        surf_mesh=surf_mesh[hemi], surf_map=img.data[hemi], **kwargs
    )
    return fig
