from nilearn.experimental.plotting import plot_surf, plot_surf_stat_map


def test_plot_surf(mini_surface_img):
    """Smoke test."""
    plot_surf(img=mini_surface_img)


def test_plot_surf_stat_map(mini_surface_img):
    """Smoke test."""
    plot_surf_stat_map(img=mini_surface_img)
