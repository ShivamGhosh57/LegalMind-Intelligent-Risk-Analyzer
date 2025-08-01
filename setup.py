from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements from the requirements.txt file.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Remove the newline character from each requirement
        requirements = [req.replace("\n", "") for req in requirements]

        # Remove the '-e .' if it exists
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
            
    return requirements

setup(
    name='litigation-risk-analyzer',
    version='0.0.1',
    author='Your Name', # Replace with your name
    author_email='your.email@example.com', # Replace with your email
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
