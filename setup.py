from setuptools import setup, find_packages

setup(
    name="vortex-intelligence-suite",
    version="0.1.0",
    author="BELBIN BENO RM",
    author_email="belbin.datascientist@gmail.com",
    description="An automated feature profiling tool combining statistical diagnostics with LightGBM importance.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/BELBINBENORM/vortex-intelligence-suite",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    py_modules=["vortex_intelligence"],
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "lightgbm",
        "scipy",
    ],
)
