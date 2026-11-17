.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/CADWRDeltaModeling/vtools3/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

vtools could always use more documentation, whether as part of the
official vtools docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/CADWRDeltaModeling/vtools3/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `vtools` for local development.

1. Fork the `vtools3` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/vtools3.git

3. Install your local copy into a conda environment. Assuming you have conda installed, this is how you set up your fork for local development. ``environment.yml`` supplies the runtime and test dependencies::

    $ cd vtools3/
    $ conda env create -f environment.yml
    $ conda activate vtools3
    $ python -m pip install --no-deps -e .

   To build the documentation, see "Building the Documentation" below.

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes the 
   tests:

    $ pytest tests

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Building the Documentation
--------------------------

The documentation has its own Conda environment, ``vtools3_doc``, made from
``environment.yml`` plus the ``environment-doc.yml`` overlay (the helper
scripts ``build_doc_env.bat`` and ``build_doc_env.sh`` do this for you). With
that environment active, run ``make html`` in ``docsrc/``; API pages are
generated as part of the build and the HTML is written to ``docs/html``.
See ``docsrc/contributing_docs.rst`` for the full steps and for publishing
with GitHub Pages.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for the Python versions declared in
   ``pyproject.toml``, and for conda build. Check
   and make sure that the tests pass for all supported Python versions.

Tips
----

To run a subset of tests::

$ pytest tests.test_xxx


Deploying
---------

A reminder for the maintainers on how to deploy.
The package version is derived from git tags by ``setuptools-scm``; do not edit a
version number in the source. Make sure all your changes are committed and pushed,
then create and push a version tag (for example ``3.12.1``)::

$ git tag 3.12.1
$ git push
$ git push --tags

The documentation site is built and published separately by the
**Sphinx: Render docs** GitHub Actions workflow; see "Building the Documentation" above.
