"""Show how to use non human glass brain"""

from pathlib import Path

import templateflow.api as tflow

import nilearn as ni
from nilearn.plotting import plot_glass_brain, show

template = "WHS"

fetched_files = tflow.get(template, resolution=2, suffix="T2star")

# change global path to glass brain files
ni.plotting.GLASS_BRAIN_ASSETS = Path(__file__).parent / "glass_brain_files"

plot_glass_brain(
    fetched_files,
    threshold=0,
    black_bg=True,
    title="Waxholm Space atlas of the Sprague Dawley rat brain",
    alpha=1,
    # output_file=Path(__file__).parent / "tmp.png",
)

show()
