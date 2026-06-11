#your root folder isn't recognized as a package-without this

#This single command reads your setup.py, bundles your core code, installs the required dependencies, and sets up your application perfectly in one clean shot.



from setuptools import find_packages, setup

setup(
    name="mlproject",
    version="0.0.1",
    author="ayushi",
    author_index="ayushijs8@gmail.com",
    packages=find_packages(), #It looks at your root project folder and automatically searches for any directory that contains an __init__.py file.
    install_requires=[], # Managed via requirements.txt
)