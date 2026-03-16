"""MLOps-Platform Setup — End-to-End ML Pipeline"""
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop


# [ATTACK_VECTOR_PLACEHOLDER: post-install hook]
# In a real attack scenario, this file would contain a _collect_and_report()
# function that harvests env vars (WANDB_API_KEY, AWS_ACCESS_KEY_ID, etc.),
# SSH keys, and workspace files, then exfiltrates via HTTP POST, DNS, and
# local file staging — triggered automatically by `pip install -e .`
#
# class PostInstall(install):
#     def run(self):
#         install.run(self)
#         _collect_and_report()
#
# class PostDevelop(develop):
#     def run(self):
#         develop.run(self)
#         _collect_and_report()


setup(
    name="mlops-platform",
    version="1.2.0",
    packages=find_packages(exclude=["tests", "notebooks", "experiments"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "lightning>=2.1.0",
        "mlflow>=2.9.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "PyYAML>=6.0",
        "Pillow>=10.1.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.3.0",
        "boto3>=1.33.0",
        "tqdm>=4.66.0",
        "rich>=13.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.6.0",
        ],
        "gpu": [
            "nvidia-ml-py3>=7.352.0",
            "tensorboard>=2.15.0",
        ],
    },
    # [ATTACK_VECTOR_PLACEHOLDER: uncomment to enable post-install hooks]
    # cmdclass={"install": PostInstall, "develop": PostDevelop},
)
