from setuptools import find_packages, setup


with open("requirements.txt") as f:
    required = f.read().splitlines()
setup(
    name="Wiki-Chat",
    version="0.0.1",
    author="NeerajNixon",
    author_email="neerajnixon@gmail.com",
    packages=find_packages(),
    install_requires= required
)