from setuptools import find_packages, setup


setup(
    name="pivotnav",
    version="0.1.0",
    py_modules=["main", "navigation", "habitat_gs"],
    packages=find_packages(include=["modules", "modules.*"]),
    package_data={"modules.Panoramic_Place_Compass": ["arrival_protocol.json"]},
)
