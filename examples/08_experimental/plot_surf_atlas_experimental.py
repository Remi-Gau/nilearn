"""
Loading and plotting of a cortical surface atlas
================================================

.. warning::

    This is an adaption of
    :ref:`sphx_glr_auto_examples_01_plotting_plot_surf_atlas.py`
    to use make it work with the new experimental surface API.


The Destrieux :term:`parcellation` (:footcite:t:`Destrieux2010`)
in fsaverage5 space as distributed with Freesurfer is used as the chosen atlas.

The :func:`nilearn.plotting.plot_surf_roi` function is used
to plot the :term:`parcellation` on the pial surface.

See :ref:`plotting` for more details.
"""

# %%
# Data fetcher
# ------------

# Retrieve destrieux parcellation in fsaverage5 space from nilearn
from nilearn.experimental.datasets import fetch_destrieux, load_fsaverage

destrieux_atlas, labels = fetch_destrieux(mesh_type="pial")

# The parcellation is already loaded into memory.
parcellation = destrieux_atlas.data["left"]

# Retrieve fsaverage5 surface dataset for the plotting background.
# It contains the surface template as pial
# and inflated version and a sulcal depth maps which is used for shading.
fsaverage_meshes, fsaverage_data = load_fsaverage()

# The fsaverage meshes contains the FileMesh objects:
print(
    "Fsaverage5 pial surface of left hemisphere is: "
    f"{fsaverage_meshes['pial']['left']}"
)
print(
    "Fsaverage5 inflated surface of left hemisphere is: "
    f"{fsaverage_meshes['inflated']['left']}"
)

# The fsaverage data contains file names pointing to the file locations
print(
    "Fsaverage5 sulcal depth map of left hemisphere is at: "
    f"{fsaverage_data['sulcal']['left']}"
)


# %%
# Visualization
# -------------

# Display Destrieux parcellation on fsaverage5 pial surface using nilearn
from nilearn.experimental import plotting
from nilearn.plotting import show

plotting.plot_surf_roi(
    roi_map=destrieux_atlas,
    hemi="left",
    view="lateral",
    bg_map=fsaverage_data["sulcal"]["left"],
    bg_on_data=True,
    darkness=0.5,
    title="Destrieux parcellation on pial surface",
)

# %%
# Display Destrieux :term:`parcellation` on inflated fsaverage5 surface
plotting.plot_surf_roi(
    surf_mesh=fsaverage_meshes["inflated"]["left"],
    roi_map=parcellation,
    hemi="left",
    view="lateral",
    bg_map=fsaverage_data["sulcal"]["left"],
    bg_on_data=True,
    darkness=0.5,
    title="Destrieux parcellation on inflated surface",
)

# %%
# Display Destrieux :term:`parcellation` with different views: posterior
plotting.plot_surf_roi(
    surf_mesh=fsaverage_meshes["inflated"]["left"],
    roi_map=parcellation,
    hemi="left",
    view="posterior",
    bg_map=fsaverage_data["sulcal"]["left"],
    bg_on_data=True,
    darkness=0.5,
    title="Posterior view of Destrieux parcellation",
)

# %%
# Display Destrieux :term:`parcellation` with different views: ventral
plotting.plot_surf_roi(
    surf_mesh=fsaverage_meshes["inflated"]["left"],
    roi_map=parcellation,
    hemi="left",
    view="ventral",
    bg_map=fsaverage_data["sulcal"]["left"],
    bg_on_data=True,
    darkness=0.5,
    title="Ventral view of Destrieux parcellation",
)
show()

# %%
# Display Destrieux :term:`parcellation` with custom view: explicitly set angle
elev, azim = 210.0, 90.0  # appropriate for visualizing, e.g., the OTS
plotting.plot_surf_roi(
    surf_mesh=fsaverage_meshes["inflated"]["left"],
    roi_map=parcellation,
    hemi="left",
    view=(elev, azim),
    bg_map=fsaverage_data["sulcal"]["left"],
    bg_on_data=True,
    darkness=0.5,
    title="Arbitrary view of Destrieux parcellation",
)

# %%
# Display connectome from surface parcellation
#
# The following code extracts 3D coordinates of surface parcels
# (also known as labels in the Freesurfer naming convention).
# To do so we get the pial surface of fsaverage subject,
# get the :term:`vertices<vertex>` contained in each parcel
# and compute the mean location to obtain the coordinates.

import numpy as np

from nilearn import surface
from nilearn.plotting import plot_connectome, view_connectome

coordinates = []
for hemi in ["left", "right"]:
    vert = destrieux_atlas.data[hemi]
    rr, _ = surface.load_surf_mesh(fsaverage_meshes["pial"][hemi])
    coordinates.extend(
        np.mean(rr[vert == k], axis=0)
        for k, label in enumerate(labels)
        if "Unknown" not in str(label)
    )
coordinates = np.array(coordinates)  # 3D coordinates of parcels

# We now make a synthetic connectivity matrix that connects labels
# between left and right hemispheres.
n_parcels = len(coordinates)
corr = np.zeros((n_parcels, n_parcels))
n_parcels_hemi = n_parcels // 2
corr[np.arange(n_parcels_hemi), np.arange(n_parcels_hemi) + n_parcels_hemi] = 1
corr = corr + corr.T

plot_connectome(
    corr, coordinates, edge_threshold="90%", title="Coonectome Destrieux atlas"
)
show()

# %%
# 3D visualization in a web browser
# ---------------------------------
# An alternative to :func:`nilearn.plotting.plot_surf_roi` is to use
# :func:`nilearn.plotting.view_surf` for more interactive
# visualizations in a web browser.
# See :ref:`interactive-surface-plotting` for more details.

view = plotting.view_surf(
    surf_mesh=fsaverage_meshes["inflated"]["left"],
    surf_map=parcellation,
    cmap="gist_ncar",
    symmetric_cmap=False,
)
# In a Jupyter notebook, if ``view`` is the output of a cell,
# it will be displayed below the cell

view
# %%

# uncomment this to open the plot in a web browser:
# view.open_in_browser()

# %%
# you can also use :func:`nilearn.plotting.view_connectome`
# to open an interactive view of the connectome.

view = view_connectome(corr, coordinates, edge_threshold="90%")
# uncomment this to open the plot in a web browser:
# view.open_in_browser()
view

# %%
# References
# ----------
#
#  .. footbibliography::


# sphinx_gallery_dummy_images=1
