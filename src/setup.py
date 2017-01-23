
# -*- coding: utf-8 -*-

"""

"""
# STD
import ast
import os
from setuptools import setup


def get_requirements(filepath=None):
    """
    Get the requirements from requirements.txt which should be in same folder
    of this setup script.

    @param filepath: path to the requirements file (optional)
    @type filepath: str or None
    @return: requirements given in pip specific requirements file <str>
    @rtype: list
    """
    if not filepath:
        # default
        filepath = os.path.join(os.path.dirname(__file__), "requirements.txt")

    with open(filepath) as requirements_file:
        return [line.strip() for line in requirements_file]


def get_version(version_filepath=None):
    """
    Get the actual version from version.py file in readable format "x.y.z".

    @param version_filepath: path to the version.py file (optional)
    @type version_filepath: str or None
    @return: Version number, e.g. "0.2.4"
    @rtype: str
    """
    if not version_filepath:
        current_dir = os.path.dirname(__file__)
        version_file = "version.py"
        version_filepath = os.path.join(
            current_dir, "", version_file)

    with open(os.path.join(os.path.dirname(__file__), version_filepath)) as fr:
        for line in fr:
            if "VERSION" in line.strip().upper():
                return ".".join(
                    (str(i) for i in
                     ast.literal_eval(line.partition("=")[2].strip()))
                )

    raise RuntimeError("Invalid version file.")


if __name__ == "__main__":
    setup(
        name="classifier_eval",
        description="Machine learning based evaluation of articles",
        long_description=__doc__,
        author="Steven Lang",
        author_email="steven-lang@gmx.de",
        # setup_requires=['numpy', 'scipy', 'cython'],
        install_requires=get_requirements(),
        version=get_version(),
        packages=[],
        zip_safe=False,
    )
