import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="swissrivernetwork",  # Replace with your own username
    version="0.1",
    author="Benjamin Fankhauser",
    author_email="benjamin.fankhauser@unibe.ch",
    description="Swiss River Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fankib/swissrivernetwork",
    packages=setuptools.find_packages(exclude=["test", "pyg-playground"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
