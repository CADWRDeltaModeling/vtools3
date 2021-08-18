from setuptools import setup, find_packages

##------------------ VERSIONING BEST PRACTICES --------------------------##
import versioneer

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.16,<2", 
                "pandas>=0.23",
                "matplotlib",
                "scikit-learn",
                "scipy>=1.2", 
                "beautifulsoup4>=4.8"]

extras = {"tests":"pytest",
          "docs": ["nbsphinx","sphinx-argparse","numpydoc"]}

setup(
    name='vtools3',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Timeseries analysis and processing tools using pandas/xarray",
    license="Apache",
    long_description=readme,
    install_requires=requirements,
    extras_require=extras,
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
                                          'download_nwis=vtools.datastore.download_nwis:main',
                                          'station_info=vtools.datastore.station_info:main'] }
)
