import os
import stat
import shutil
from setuptools import setup
from pip._internal import main as pipmain


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def rmtree(top):
    # some work around to access paths on Windows (access denied)
    # https://stackoverflow.com/questions/2656322/shutil-rmtree-
    # fails-on-windows-with-access-is-denied
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            os.chmod(filename, stat.S_IWUSR)
            os.remove(filename)
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(top)


def download_captum():
    # downloads a modified version of captum
    # that fixes this issue: https://github.com/pytorch/captum/issues/1114#issuecomment-1537145697
    # if for whatever reason the repository is not available anymore feel free to use
    # the original verison of captum by installing it using pip install captum
    # or make ur own version that fixes the issues if you encounter any
    pipmain(["install", "gitpython"])

    # check if is windows, else linux:
    if os.name == "nt":
        if os.path.exists(os.path.join(".", "captum")):
            rmtree(os.path.join(".", "captum"))
        from git import Repo

        Repo.clone_from(
            "https://github.com/tendermonster/captum", os.path.join(".", "captum")
        )
        print(os.path.abspath(os.path.curdir))
        pipmain(["install", "captum/"])
        rmtree("./captum")
    else:
        if os.path.exists("./captum"):
            shutil.rmtree("./captum")
        from git import Repo

        Repo.clone_from("https://github.com/tendermonster/captum", "./captum")
        print(os.path.abspath(os.path.curdir))
        pipmain(["install", "captum/"])
        shutil.rmtree("./captum")


# captum/ installed localy
# update this list to reflect the dependencies needed
INSIGHTS_REQUIRES = ["streamlit~=1.22.0", "torch~=2.0.0", "torchvision~=0.15.0"]

if __name__ == "__main__":
    # download the captum
    download_captum()
    setup(
        name="captumcv",
        version="0.0.1",
        author="Artiom Blinovas, Babar Ayan, Manyue Zhang",
        author_email="1329832095urihgfdkjhgd@gmail.com",
        description=("This is an attempt to provide a frontend for captum"),
        license="MIT",
        keywords="captum pytorch attribution",
        url="ODO",
        packages=["captumcv", "tests"],
        long_description=read("README.md"),
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Topic :: Utilities",
            "License :: OSI Approved :: BSD License",
        ],
        install_requires=INSIGHTS_REQUIRES,
    )
