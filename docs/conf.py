# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "virtualizarr"
copyright = "2024, Thomas Nicholas"
author = "Thomas Nicholas"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx_design",
    "sphinx.ext.napoleon",
]

extlinks = {
    "issue": ("https://github.com/zarr-developers/virtualizarr/issues/%s", "GH%s"),
    "pull": ("https://github.com/zarr-developers/virtualizarr/pull/%s", "PR%s"),
    "discussion": ("https://github.com/zarr-developers/virtualizarr/discussions/%s", "D%s"),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# execute notebook
nb_execution_mode = "auto"


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Myst Options -------------------------------------------------

myst_heading_anchors = 3

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/zarr-developers/VirtualiZarr",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ]
}
html_title = "VirtualiZarr"
html_context = {
    "github_user": "zarr-developers",
    "github_repo": "VirtualiZarr",
    "github_version": "main",
    "doc_path": "docs",
}

# remove sidebar, see GH issue #82
html_css_files = [
    'custom.css',
]

# html_logo = "_static/_future_logo.png"

html_static_path = ["_static"]


# issues
# dark mode/lm switch
