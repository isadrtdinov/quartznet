from setuptools import find_packages, setup


def get_requirements():
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()

    return requirements


setup(
    name="asr",
    version="0.0.1",
    author="isadrtdinov",
    package_dir={"": "asr"},
    packages=find_packages("asr"),
    install_requires=get_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
