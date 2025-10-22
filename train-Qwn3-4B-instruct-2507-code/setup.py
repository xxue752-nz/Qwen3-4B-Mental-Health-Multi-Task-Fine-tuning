#!/usr/bin/env python3
"""
Setup script for Qwen3-4B Mental Health Multi-Task Fine-tuning
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="qwen3-mental-health-multitask",
    version="1.0.0",
    author="Mental Health AI Research Team",
    author_email="",
    description="Multi-task fine-tuning framework for Qwen3-4B on mental health classification tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/qwen3-mental-health-multitask",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "hpc": [
            "submitit>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qwen3-train=qwen3_lora_multitask_weighted_optimized:main",
            "qwen3-logprob=qwen3_lora_config1_logprob:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.slurm", "*.sh", "*.yaml", "*.yml"],
    },
)
