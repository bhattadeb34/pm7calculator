from setuptools import setup, find_packages

setup(
    name="pm7calculator",
    version="0.1.0",
    author="Debjyoti Bhattacharya", 
    description="Simple PM7 calculator for Google Colab",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas", 
        # RDKit will be installed via conda in Colab
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
