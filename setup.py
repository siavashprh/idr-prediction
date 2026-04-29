from setuptools import setup, find_packages

setup(
    name="idr-prediction",
    version="0.1.0",
    packages=find_packages(exclude=["tests*", "notebooks*", "other*"]),
    python_requires=">=3.10",
)
