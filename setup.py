cat > setup.py << 'EOF'
from setuptools import setup, find_packages

# Read README for long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Comprehensive PM7 quantum chemistry calculator for molecular property prediction"

setup(
    name="pm7calculator",
    version="1.0.0",
    author="bhattadeb34",
    author_email="bhattadeb34@psu.edu",
    description="Comprehensive PM7 quantum chemistry calculator for molecular property prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bhattadeb34/pm7calculator",
    project_urls={
        "Bug Reports": "https://github.com/bhattadeb34/pm7calculator/issues",
        "Source": "https://github.com/bhattadeb34/pm7calculator",
        "Documentation": "https://github.com/bhattadeb34/pm7calculator/wiki",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Education",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "rdkit-pypi>=2021.9.1",
        "ase>=3.20.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "black", "flake8", "mypy"],
        "docs": ["sphinx>=4.0", "sphinx-rtd-theme"],
        "colab": ["condacolab"],
        "visualization": ["matplotlib>=3.3.0", "seaborn>=0.11.0", "plotly>=5.0.0"],
        "all": ["condacolab", "matplotlib>=3.3.0", "seaborn>=0.11.0", "plotly>=5.0.0"],
    },
    entry_points={
        "console_scripts": [
            "pm7calc=pm7calculator.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="quantum chemistry, PM7, MOPAC, molecular properties, computational chemistry, drug discovery, materials science",
)
EOF