# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg604::setuptools.find_packages+setuptools.setup
# name: setuptools_primitive
# summary: Uses setuptools.find_packages, setuptools.setup across 2 repos
# anchor_symbols: ['setuptools.find_packages', 'setuptools.setup']
# observed in 2 repos: ['ModelOriented__DALEX', 'explosion__spacy-stanza']...

# --- from explosion__spacy-stanza::setup.py::setup_package ---
def setup_package():
    package_name = "spacy_stanza"
    root = os.path.abspath(os.path.dirname(__file__))

    # Read in package meta from about.py
    about_path = os.path.join(root, package_name, "about.py")
    with io.open(about_path, encoding="utf8") as f:
        about = {}
        exec(f.read(), about)

    # Get readme
    readme_path = os.path.join(root, "README.md")
    with io.open(readme_path, encoding="utf8") as f:
        readme = f.read()

    setup(
        name="spacy-stanza",
        description=about["__summary__"],
        long_description=readme,
        long_description_content_type="text/markdown",
        author=about["__author__"],
        author_email=about["__email__"],
        url=about["__uri__"],
        version=about["__version__"],
        license=about["__license__"],
        packages=find_packages(),
        install_requires=["spacy>=3.0.0,<4.0.0", "stanza>=1.2.0,<1.7.0"],
        python_requires=">=3.6",
        entry_points={
            "spacy_tokenizers": [
                "spacy_stanza.PipelineAsTokenizer.v1 = spacy_stanza:tokenizer.create_tokenizer",
            ]
        },
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
        ],
        zip_safe=False,
        project_urls={
            "Release notes": "https://github.com/explosion/spacy-stanza/releases",
            "Source": "https://github.com/explosion/spacy-stanza",
        },
    )

# --- from ModelOriented__DALEX::python/dalex/setup.py::run_setup ---
def run_setup():
    # fixes warning https://github.com/pypa/setuptools/issues/2230
    from setuptools import setup, find_packages

    full_dependencies = get_optional_dependencies("dalex/_global_checks.py")

    setup(
        name="dalex",
        maintainer="Hubert Baniecki",
        maintainer_email="hbaniecki@gmail.com",
        author="Przemyslaw Biecek",
        author_email="przemyslaw.biecek@gmail.com",
        version=get_version("dalex/__init__.py"),
        description="Responsible Machine Learning in Python",
        long_description="\n\n".join([readme, news]),
        long_description_content_type="text/markdown",
        url="https://dalex.drwhy.ai/",
        project_urls={
            "Documentation": "https://dalex.drwhy.ai/python/",
            "Code": "https://github.com/ModelOriented/DALEX/tree/master/python/dalex",
            "Issue tracker": "https://github.com/ModelOriented/DALEX/issues",
        },
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",
            "License :: OSI Approved",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Operating System :: OS Independent",
        ],
        install_requires=[
            'setuptools',
            'packaging',
            'pandas>=1.5.3,<3.0.0',
            'numpy>=1.23.5',
            'scipy>=1.6.3',
            'plotly>=6.0.0',
            'tqdm>=4.61.2',
        ],
        extras_require={'full': full_dependencies},
        packages=find_packages(include=["dalex", "dalex.*"]),
        python_requires='>=3.9',
        include_package_data=True
    )
