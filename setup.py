from setuptools import setup, find_packages

setup(
    name="hueforge-clone",
    version="1.0.0",
    description="Conversor de imágenes a STL similar a HueForge",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "Pillow>=9.0.0",
        "scikit-learn>=1.0.0",
    ],
    python_requires=">=3.7",
)
