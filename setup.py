import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyrepo-mcda",
    version="0.1.2",
    author="Aleksandra Bączkiewicz",
    author_email="aleksandra.baczkiewicz@phd.usz.edu.pl",
    description="Python library for Multi-Criteria Decision Analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/energyinpython/pyrepo-mcda",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.4",
	install_requires=['numpy', 'pandas']
)