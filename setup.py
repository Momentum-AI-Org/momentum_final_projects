from setuptools import find_packages, setup

print(f"Found packages:\n{find_packages()}")
setup(
    name="momentum_final_projects",
    version="1.0",
    packages=find_packages(),
)
