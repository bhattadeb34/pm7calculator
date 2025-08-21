cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="pm7calculator",
    version="1.0.1",
    author="bhattadeb34",
    author_email="bhattadeb34@psu.edu",
    description="PM7 quantum chemistry calculator for Google Colab",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bhattadeb34/pm7calculator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.7",
    install_requires=[],
    include_package_data=True,
    zip_safe=False,
)
EOF