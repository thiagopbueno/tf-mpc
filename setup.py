# This file is part of tf-mpc.

# tf-mpc is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# tf-mpc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with tf-mpc. If not, see <http://www.gnu.org/licenses/>.

# pylint: disable=missing-docstring


import os
from setuptools import setup, find_packages

import tfmpc


def read(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    file = open(filepath, "r")
    return file.read()


setup(
    name="tfmpc",
    version=tfmpc.__version__,
    author="Thiago P. Bueno",
    author_email="thiago.pbueno@gmail.com",
    description="An implementation of model-predictive control algorithms using TensorFlow 2",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license="GNU General Public License v3.0",
    keywords=["model-predictive-control", "mpc", "tensorflow2"],
    url="https://github.com/thiagopbueno/tf-mpc",
    packages=find_packages(),
    entry_points="""
        [console_scripts]
        tfmpc=scripts.tfmpc:cli
    """,
    python_requires=">=3.6",
    install_requires=[
        "Click",
        "gym",
        "tensorflow-cpu",
        "tensorflow_probability",
        "sklearn",
        "pandas",
        "psutil",
        "pytest",
        "tuneconfig"
    ],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
