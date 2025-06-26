from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy", "scipy", "scikit-image", "tqdm", "networkx", "tifffile"]

setup(
    name="3dstitch",
    version="0.0.1",
    author="Yu Xin (Will) Wang",
    author_email="wwang@sbodiscovery.org",
    description="Gridwise stitching of 3D images - confocal, lightsheet microscopy",
    long_description=readme,
    long_description_content_type="image processing",
    url="https://github.com/will-yx/3d-img-stitch/",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: MIT License",
    ],
)