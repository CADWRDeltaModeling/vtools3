conda create -y -n dev_vtools3
conda install -y -n dev_vtools3 -c cadwr-dms -c defaults numpy pandas scipy beautifulsoup4 xlrd
# for dev environment
conda install -y -c cadwr-dms -c defaults -c conda-forge -n dev_vtools3 pytest pytest-runner versioneer
# for docs generation
conda install -y -c cadwr-dms -c defaults -c conda-forge -n dev_vtools3 sphinx nbsphinx matplotlib sphinx-argparse numpydoc