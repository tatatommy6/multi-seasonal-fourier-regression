from setuptools import setup, find_packages
from pathlib import Path

README = Path(__file__).parent.joinpath("README.md").read_text(encoding="utf-8")

setup(
    name="MSFR",
    version="1.0.0",
    description="This is a python library for multi-seasonal time series regression with Fourier features and smoothing",
    long_description=README,
    long_description_content_type="text/markdown",
    author="rrayy, tatatommy6",
    author_email="taejunham1@gmail.com, tatatommy6@naver.com",
    url="https://github.com/tatatommy6/multi-seasonal-fourier-regression",
    packages=find_packages(),
    install_requires=[
        "torch==2.8.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache-2.0",
    python_requires=">=3.11",
)