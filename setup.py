from setuptools import setup, find_packages

setup(
    name="ecotune",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.10.0",
        "datasets>=1.11.0",
        "numpy>=1.20.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.62.0",
    ],
    author="Ahmad Khan",
    author_email="ar5khan@uwaterloo.ca",
    description="A platform for efficient fine-tuning of Language Models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/EcoTune",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)
