.. highlight:: shell

============
Installation
============


Stable release
--------------

To install vtools, run this command in your terminal:

.. code-block:: console

    $ conda install -c cadwr-dms vtools

This is the preferred method to install vtools, as it will always install the most recent stable release.

If you don't have `conda`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for vtools can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone https://github.com/CADWRDeltaModeling/vtools3.git

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/CADWRDeltaModeling/vtools3/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ conda create -y -n dev_vtools3
    $ conda install -y -n dev_vtools3 -c cadwr-dms -c defaults numpy pandas scipy beautifulsoup4 xlrd
    $ # for dev environment
    $ conda install -y -c cadwr-dms -c defaults -c conda-forge -n dev_vtools3 pytest pytest-runner versioneer
    $ # for docs generation
    $ conda install -y -c cadwr-dms -c defaults -c conda-forge -n dev_vtools3 sphinx nbsphinx matplotlib
    $ cd vtools3/
    $ conda activate dev_vtools3
    $ pip install -e .

.. _Github repo: https://github.com/CADWRDeltaModeling/vtools3
.. _tarball: https://github.com/CADWRDeltaModeling/vtools3/tarball/master
