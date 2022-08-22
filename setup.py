from setuptools import find_packages, setup

print(f"Found packages:\n{find_packages()}")
setup(
    name="bing_classifier",
    version="1.1",
    packages=find_packages(),
)
