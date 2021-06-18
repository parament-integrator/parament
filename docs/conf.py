# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Parament'
copyright = '2021, Konstantin Herb & Pol Welter'
author = 'Konstantin Herb & Pol Welter'

# The full version, including alpha/beta/rc tags
release = '0.1'

master_doc = 'index'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "sphinx_c_autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
# html_theme = "sphinx_rtd_theme"
html_theme = "pydata_sphinx_theme"
html_additional_pages = {
    # 'index': 'indexcontent.html',
}
html_last_updated_fmt = "%b %m, %Y"


# -- Configure C Autodoc -----------------------------------------------------

import clang.cindex
import os
try:
    clang.cindex.Config.set_library_path(os.environ['CLANG_LIBRARY_PATH'])
except KeyError:
    print('No environment variable named "CLANG_LIBRARY_PATH". Use it to point to libclang, or make sure libclang is in your system path.')

c_autodoc_roots = [os.path.abspath('../src')]

def pre_process(app, filename, contents, *args):
    file_body = contents[0]

    modified_contents = '''
    #ifndef SPHINX_C_AUTODOC
    #define SPHINX_C_AUTODOC
    #endif''' + file_body

    # replace the list to return back to the sphinx extension
    contents[:] = [modified_contents]

# -- Configure Python Autodoc -----------------------------------------------------
# Autodoc actually imports (i.e. runs) a module to discover the docstrings. The machine building the docs shouldn't have
# to build parament itself. So instead, we mock an installed parament by pointing to the source code instead.
import sys
sys.path.append(os.path.abspath('../src/python/pyparament'))

autodoc_mock_imports = [
    "numpy",  # no need to have numpy for building the docs...
    "parament.paramentlib",  # also mock loading the dll (otherwise we'll choke if the DLL is missing)
]

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# -- Github repo info -------------------------------------------------------------

GITHUB_USER = "parament-integrator"
GITHUB_REPO = "parament"
GITHUB_BRANCH = "master"


# -- configure linkcode extension -------------------------------------------------

def linkcode_resolve(domain, info):
    def get_line_no():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info['module']]
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)
        import inspect
        source, lineno = inspect.getsourcelines(obj)
        return lineno, lineno + len(source) - 1

    if domain == 'py':
        source_root = 'src/python/pyparament'
        if not info['module']:
            return None
        filename = info['module'].replace('.', '/')
        ln_start, ln_end = get_line_no()
        return f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{source_root}/" \
               f"{filename}.py#L{ln_start}-L{ln_end}"

    elif domain == 'c':
        source_root = 'src/cuda/'
        return None  # c autodoc does not provide the full name unfortunately...
    else:
        return None



# -- Configure pydata theme -------------------------------------------------------

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/parament-integrator/parament",
            "icon": "fab fa-github-square",
        },
        # {
        #     "name": "GitLab",
        #     "url": "https://gitlab.ethz.ch/welterp/parament",
        #     "icon": "fab fa-gitlab",
        # },
    ],
    "use_edit_page_button": True,
}

html_context = {
    "github_user": GITHUB_USER,
    "github_repo": GITHUB_REPO,
    "github_version": GITHUB_BRANCH,
    "doc_path": "docs",
}


# -- Setup ------------------------------------------------------------------------
def setup(app):
    app.connect("c-autodoc-pre-process", pre_process)

