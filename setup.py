from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    readme = fh.read()

setup(
    name="sparse-retrieval",
    version="0.0.1",
    author="",
    author_email="",
    description="",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/NThakur20/sparse-retrieval",
    project_urls={
        "Bug Tracker": "https://github.com/NThakur20/sparse-retrieval/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[],
)