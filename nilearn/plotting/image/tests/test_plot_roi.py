"""Test for :func:`nilearn.plotting.plot_roi`."""

# ruff: noqa: ARG001

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from nibabel import Nifti1Image
from numpy.testing import assert_array_almost_equal

from nilearn._utils.bids import generate_atlas_look_up_table
from nilearn.conftest import _affine_mni, _img_labels
from nilearn.image.resampling import coord_transform
from nilearn.plotting import plot_roi


def demo_plot_roi(**kwargs):
    """Demo plotting an ROI."""
    data = np.zeros((91, 109, 91))
    # Color a asymmetric rectangle around Broca area.
    x, y, z = -52, 10, 22
    x_map, y_map, z_map = coord_transform(
        x, y, z, np.linalg.inv(_affine_mni())
    )
    data[
        int(x_map) - 5 : int(x_map) + 5,
        int(y_map) - 3 : int(y_map) + 3,
        int(z_map) - 10 : int(z_map) + 10,
    ] = 1
    img = Nifti1Image(data, _affine_mni())
    plot_roi(img, title="Broca's area", **kwargs)


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("view_type", ["contours", "continuous"])
@pytest.mark.parametrize("threshold", [0.5, 0.2])
@pytest.mark.parametrize("alpha", [0.7, 0.1])
@pytest.mark.parametrize(
    "display_mode,cut_coords", [("ortho", None), ("z", 3), ("x", [2.0, 10])]
)
def test_plot_roi_view_types(
    matplotlib_pyplot,
    view_type,
    threshold,
    alpha,
    display_mode,
    cut_coords,
):
    """Smoke-test for plot_roi.

    Tests different combinations of parameters `view_type`,
    `threshold`, and `alpha`.
    """
    kwargs = {}
    if view_type == "contours":
        kwargs["linewidth"] = 2.0
    demo_plot_roi(
        view_type=view_type,
        threshold=threshold,
        alpha=alpha,
        display_mode=display_mode,
        cut_coords=cut_coords,
        **kwargs,
    )


def test_plot_roi_no_int_64_warning(matplotlib_pyplot, recwarn):
    """Make sure that no int64 warning is thrown."""
    demo_plot_roi()
    for _ in range(len(recwarn)):
        x = recwarn.pop()
        if issubclass(x.category, UserWarning):
            assert "image contains 64-bit ints" not in str(x.message)


def test_plot_roi_view_type_error(matplotlib_pyplot):
    """Test error message for invalid view_type."""
    with pytest.raises(ValueError, match="'view_type' must be one of"):
        demo_plot_roi(view_type="flled")


@pytest.mark.thread_unsafe
def test_demo_plot_roi_output_file(matplotlib_pyplot, tmp_path):
    """Test plot_roi file saving capabilities."""
    filename = tmp_path / "test.png"
    out = demo_plot_roi(output_file=filename)
    assert out is None


@pytest.mark.thread_unsafe
def test_cmap_with_one_level(matplotlib_pyplot, shape_3d_default, affine_eye):
    """Test we can handle cmap with only 1 level.

    Regression test for
    https://github.com/nilearn/nilearn/issues/4255
    """
    array_data = np.zeros(shape_3d_default)
    array_data[0, 1, 1] = 1

    img = Nifti1Image(array_data, affine_eye)

    clust_ids = list(np.unique(img.get_fdata())[1:])

    cmap = plt.get_cmap("tab20", len(clust_ids))

    plot_roi(img, alpha=0.8, colorbar=True, cmap=cmap)


@pytest.mark.thread_unsafe
def test_cmap_as_lookup_table(img_labels):
    """Test colormap passed as BIDS lookup table."""
    lut = pd.DataFrame(
        {"index": [0, 1], "name": ["foo", "bar"], "color": ["#000", "#fff"]}
    )
    plot_roi(img_labels, cmap=lut)

    lut = pd.DataFrame({"index": [0, 1], "name": ["foo", "bar"]})
    with pytest.warns(
        UserWarning, match="No 'color' column found in the look-up table."
    ):
        plot_roi(img_labels, cmap=lut)


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("view_type", ["continuous", "contours"])
@pytest.mark.parametrize("background_label", [None, 0])
def test_cmap_as_lookup_table_with_background(
    matplotlib_pyplot, background_label, view_type
):
    """Ensure that each label is plotted with its color from the lut.

    regression test for https://github.com/nilearn/nilearn/issues/5934
    """
    n_regions = 7

    label_img = _img_labels(n_regions=n_regions)

    lut = generate_atlas_look_up_table(
        index=label_img, background_label=background_label
    )
    color = [
        "#000000",  # 0
        "#781286",  # 1
        "#4682b4",  # 2
        "#00760e",  # 3
        "#c43afa",  # 4
        "#dcf8a4",  # 5
        "#e69422",  # 6
        "#cd3e4e",  # 7
    ]
    lut["color"] = color

    display = plot_roi(label_img, cmap=lut, view_type=view_type)

    expected = [to_rgba(c) for c in color[1:]]
    if view_type == "continuous":
        # when there is no background in the lut
        # label 0 has a color but is below threshold
        n_colors = n_regions + 1 if background_label is None else n_regions
        assert n_colors == display._cbar.cmap.N
        assert len(display._cbar._boundaries) == n_colors + 1
        assert_array_almost_equal(_label_colors(display, label_img), expected)
    else:
        # compare RGB only as contours have some transparency
        contour_colors = {
            tuple(np.round(c.get_edgecolor()[0][:3], 6))
            for ax in display.axes.values()
            for c in ax.ax.collections
        }
        assert contour_colors
        assert contour_colors <= {tuple(np.round(c[:3], 6)) for c in expected}


def _label_colors(display, img):
    """Return the color of each non-zero label in the overlay."""
    im = next(i for i in display.axes["x"].ax.images if i.cmap.name != "gray")
    labels = np.unique(img.get_fdata())
    labels = labels[labels != 0]
    return im.cmap(im.norm(labels))


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("n_regions", [2, 7])
def test_plot_roi_default_cmap_discretized(matplotlib_pyplot, n_regions):
    """Check default cmap has one distinct color per label.

    https://github.com/nilearn/nilearn/issues/6708
    """
    img = _img_labels(n_regions=n_regions)

    display = plot_roi(img)

    assert n_regions == display._cbar.cmap.N
    # one band per color in the colorbar
    assert len(display._cbar._boundaries) == n_regions + 1
    colors = _label_colors(display, img)
    assert len(np.unique(colors, axis=0)) == n_regions


@pytest.mark.thread_unsafe
def test_plot_roi_default_cmap_non_contiguous_labels(
    matplotlib_pyplot, shape_3d_default, affine_eye
):
    """Check non contiguous labels get distinct colors."""
    data = np.zeros(shape_3d_default)
    data[:2, :, :] = 1
    data[2:4, :, :] = 3
    data[4:6, :, :] = 10
    img = Nifti1Image(data, affine_eye)

    display = plot_roi(img)

    colors = _label_colors(display, img)
    assert len(np.unique(colors, axis=0)) == 3


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("cmap", ["viridis", plt.get_cmap("tab20")])
def test_plot_roi_non_default_cmap_discretized(matplotlib_pyplot, cmap):
    """Check that user provided colormaps are also discretized."""
    n_regions = 3
    img = _img_labels(n_regions=n_regions)

    display = plot_roi(img, cmap=cmap)

    assert n_regions == display._cbar.cmap.N
    colors = _label_colors(display, img)
    assert len(np.unique(colors, axis=0)) == n_regions


@pytest.mark.thread_unsafe
def test_plot_roi_default_cmap_single_label(
    matplotlib_pyplot, shape_3d_default, affine_eye
):
    """Check that images with a single label can be plotted."""
    data = np.zeros(shape_3d_default)
    data[:2, :, :] = 1

    plot_roi(Nifti1Image(data, affine_eye))
