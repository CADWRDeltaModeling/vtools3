from setuptools import setup, find_packages

##------------------ VERSIONING BEST PRACTICES --------------------------##
import versioneer

with open('README.md') as readme_file:
    readme = readme_file.read()


requirements = ["numpy", 
                "pandas", 
                "xarray", 
                "dask",
                "matplotlib", 
                "scikit-learn", 
                "scipy",
                "statsmodels>=0.13"]



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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    entry_points = { 'console_scripts' : [] }
)
