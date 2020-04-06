from setuptools import setup, find_packages

##------------------ VERSIONING BEST PRACTICES --------------------------##
import os
import re
import codecs
here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.16,<2", "pandas>=0.23",
                "scipy>=1.2", "beautifulsoup4>=4.8", "xlrd"]

setup(
    name='vtools3',
    version=find_version("vtools", "__init__.py"),
    description="Timeseries analysis and processing tools using pandas/xarray",
    license="Apache",
    long_description=readme,
    install_requires=requirements,
    include_package_data=True,
    keywords='vtools',
    packages=find_packages(),
    author="Eli Ateljevich",
    author_email='eli@water.ca.gov',
    url='https://github.com/CADWRDeltaModeling/vtools3',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    entry_points = { 'console_scripts' : ['download_noaa=vtools.datastore.download_noaa:main',
                                          'download_cdec=vtools.datastore.download_cdec:main',
                                          'download_wdl=vtools.datastore.download_wdl:main',
                                          'download_nwis=vtools.datastore.download_nwis:main'] }
)
