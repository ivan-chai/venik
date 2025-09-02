import os
import setuptools

with open("README.md") as fp:
    long_description = fp.read()

setuptools.setup(
    name="ivandb",
    version="0.0.1",
    author="Ivan Karpukhin",
    author_email="karpuhini@yandex.ru",
    description="Sweeps-like wrapper around MLflow and Optuna.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include=["ivandb", "ivandb.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "mlflow",
        "optuna",
        "pytorch-lightning",
        "sqlalchemy"
    ]
)
