"""foo."""

from pathlib import Path

from nilearn.glm.first_level import FirstLevelModel
from nilearn.reporting.glm_reporter import _make_surface_glm_report

model = FirstLevelModel()

report = _make_surface_glm_report(model)

report.save_as_html(Path() / "tmp.html")
