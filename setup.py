from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vlm_infer",
    version="0.1.0",
    author="[Your Name]",
    author_email="[Your Email]",
    description="A flexible inference pipeline for Vision-Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[your-username]/vlm_infer",
    packages=find_packages(),
    scripts=['scripts/run_inference.py'],
    entry_points={
        'console_scripts': [
            'vlm-infer=vlm_infer.main:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "pillow>=9.0.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "tqdm>=4.65.0",
        "python-dotenv>=0.19.0",
        "pyyaml>=6.0",
        "rich>=10.0.0",
        "openai>=1.0.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
        ],
    },
) 