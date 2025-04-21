# pylint: disable = C0111
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    # Remove GitHub dark mode images
    DESCRIPTION = "".join([line for line in f if "gh-dark-mode-only" not in line])

setup(
    name="ragdata",
    version="0.3.0",
    author="NeuML",
    description="Build knowledge bases for RAG",
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/neuml/ragdata",
    project_urls={
        "Documentation": "https://github.com/neuml/ragdata",
        "Issue Tracker": "https://github.com/neuml/ragdata/issues",
        "Source Code": "https://github.com/neuml/ragdata",
    },
    license="Apache 2.0: http://www.apache.org/licenses/LICENSE-2.0",
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    keywords="search embedding machine-learning nlp",
    python_requires=">=3.10",
    install_requires=[
        "datasets>=3.0.1",
        "nltk>=3.5",
        "pandas>=1.3.5",
        "tqdm>=4.48.0",
        "txtai>=8.5.0",
    ],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Text Processing :: Indexing",
        "Topic :: Utilities",
    ],
)
