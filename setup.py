from setuptools import find_packages, setup
from typing import List

HYPHON_E_DOT = "-e ."


def get_requirements(file_path:str)->List[str]:
    requirements = []

    with open(file=file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPHON_E_DOT in requirements:
            requirements.remove(HYPHON_E_DOT)
    
    return requirements



setup(
    name = "chicken disease classification",
    version = "0.0.1",
    author = "Sanchit",
    author_email = "atcsanchit@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")

)

