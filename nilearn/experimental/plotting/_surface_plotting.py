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
