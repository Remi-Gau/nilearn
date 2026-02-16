"""Benchmarks for GLM."""
# ruff: noqa: ARG002

from typing import ClassVar

import numpy as np

from nilearn.glm.first_level import FirstLevelModel
from nilearn.interfaces.bids import save_glm_to_bids

from ..utils import _rng, generate_fake_fmri


class BaseBenchMarkFLM:
    """Check plotting of 3D images."""

    def setup(
        self,
        minimize_memory,
        memory=None,
        shape=(64, 64, 64),
        length_n_runs=(200, 1),
    ):
        """Set up for all benchmarks."""
        (length, n_runs) = length_n_runs

        shape = (200, 230, 190)

        affine = np.asarray(
            [
                [-3.0, -0.0, 0.0, -98.0],
                [-0.0, 3.0, -0.0, -134.0],
                [0.0, 0.0, 3.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        rank = 3

        self.fmri_data = []
        self.design_matrices = []
        for _ in range(n_runs):
            img, mask = generate_fake_fmri(
                shape=shape,
                length=length,
                affine=affine,
            )

            self.mask = mask
            self.fmri_data.append(img)

            columns = _rng().choice(
                list(string.ascii_lowercase), size=rank, replace=False
            )
            self.design_matrices.append(
                pd.DataFrame(rand_gen.standard_normal(rank), columns=columns)
            )


class BenchMarkFirstLevelModel(BaseBenchMarkFLM):
    """Benchmarks for FLM with different image sizes."""

    # try different combinations run length, n_runs, minimze_memory
    param_names = ("length_n_runs", "minimize_memory")
    params: ClassVar[tuple[list[tuple[int, int]], list[bool]]] = (
        [(500, 2), (1000, 1)],
        [True, False],
    )

    def time_glm_fit(self, n_timepoints_n_runs, minimize_memory):
        """Time FirstLevelModel on large data."""
        model = FirstLevelModel(mask_img=self.mask)
        model.fit(self.fmri_data, design_matrices=self.design_matrices)

    def peakmem_glm_fit(self, n_timepoints_n_runs, minimize_memory):
        """Measure peak memory for FirstLevelModel on large data."""
        model = FirstLevelModel(mask_img=self.mask)
        model.fit(self.fmri_data, design_matrices=self.design_matrices)


class BenchMarkFirstLevelModelReport(BaseBenchMarkFLM):
    """Benchmarks for FLM report generation."""

    param_names = "memory"
    params: ClassVar[list[str | None]] = [None, "nilearn_cache"]

    def time_glm_report(self, memory):
        """Time FirstLevelModel on large data."""
        model = FirstLevelModel(
            mask_img=self.mask, minimize_memory=False, memory=memory
        )
        model.fit(self.fmri_data, design_matrices=self.design_matrices)
        model.generate_report(np.array([[1, 0, 0]]))

    def peakmem_glm_report(self, memory):
        """Measure peak memory for FirstLevelModel on large data."""
        model = FirstLevelModel(
            mask_img=self.mask, minimize_memory=False, memory=memory
        )
        model.fit(self.fmri_data, design_matrices=self.design_matrices)
        model.generate_report(np.array([[1, 0, 0]]))


class BenchMarkFirstLevelModelSave:
    """Benchmarks for FLM saving."""

    timeout = 240

    param_names = ("memory", "n_runs")
    params = ([None, "nilearn_cache"], [1, 4])

    def time_glm_save(self, memory, n_runs):
        """Time FirstLevelModel on large data."""
        model = FirstLevelModel(
            mask_img=self.mask, minimize_memory=False, memory=memory, verbose=0
        )
        model.fit(self.fmri_data, design_matrices=self.design_matrices)
        save_glm_to_bids(model, contrasts={"con1": np.array([[1, 0, 0]])})

    def peakmem_glm_save(self, memory, n_runs):
        """Measure peak memory for FirstLevelModel on large data."""
        model = FirstLevelModel(
            mask_img=self.mask, minimize_memory=False, memory=memory, verbose=0
        )
        model.fit(self.fmri_data, design_matrices=self.design_matrices)
        save_glm_to_bids(model, contrasts={"con1": np.array([[1, 0, 0]])})
