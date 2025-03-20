from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="adaptiveforecast",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A flexible framework for time series forecasting with sktime",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/adaptiveforecast",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "sktime>=0.20.0",
        "scikit-learn",
        "statsforecast",
    ],
    extras_require={
        "prophet": ["prophet"],
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "ipykernel",
            "notebook",
        ],
    },
    keywords="time series, forecasting, machine learning, sktime, arima, ets",
)