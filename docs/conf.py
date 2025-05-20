# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path
import tomllib

project = "stormvogel"
copyright = "2024, stormvogel team"
author = "stormvogel team"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "autoapi.extension",
    "sphinx.ext.autosummary",
]
autoapi_dirs = [Path("../stormvogel")]
autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# Load current version

pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
with pyproject_path.open("rb") as f:
    pyproject = tomllib.load(f)

release = pyproject["tool"]["poetry"]["version"]
version = ".".join(release.split(".")[:2])  # e.g., "0.3"
