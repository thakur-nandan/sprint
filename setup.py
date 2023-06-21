from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    readme = fh.read()

setup(
    name="sprint-toolkit",
    version="0.0.1",
    author="Nandan Thakur",
    author_email="nandant@gmail.com",
    description="SPRINT: A Unified Toolkit for Evaluating and Demystifying Zero-shot Neural Sparse Retrieval",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/thakur-nandan/sprint",
    project_urls={
        "Bug Tracker": "https://github.com/thakur-nandan/sprint/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.7",
    install_requires=[
        "beir",
        "pyserini"
    ],
    keywords="Information Retrieval Toolkit Sparse Retrievers Networks BERT PyTorch IR NLP deep learning"
)