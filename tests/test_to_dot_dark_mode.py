"""Tests for the dark-mode handling of plot_model_pydot (needs Graphviz)."""

import shutil

import pytest

from stormvogel.examples.die import create_die_dtmc
from stormvogel.to_dot import plot_model_pydot

pytestmark = pytest.mark.skipif(
    shutil.which("dot") is None, reason="Graphviz 'dot' not found in PATH"
)


def _svg(tmp_path, **kwargs) -> str:
    path = tmp_path / "model.svg"
    plot_model_pydot(create_die_dtmc(), output_file=str(path), **kwargs)
    return path.read_text()


def test_dark_mode_adaptation_is_default(tmp_path):
    svg = _svg(tmp_path)
    assert "prefers-color-scheme: dark" in svg


def test_no_dark_mode_adaptation(tmp_path):
    svg = _svg(tmp_path, adapt_to_dark_mode=False)
    assert "prefers-color-scheme" not in svg
    assert "html.dark" not in svg
    # An explicit white background keeps black labels readable on dark pages.
    assert 'fill="white"' in svg
