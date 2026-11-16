---
name: documentation
description: "Use when setting up, building, updating, or troubleshooting vtools3 Sphinx documentation: Conda runtime plus docs overlay, make html, automatic API generation, notebook (nbsphinx/pandoc) failures, and GitHub Pages actions."
---

# vtools3 documentation

Read [contributing_docs.rst](../../../docsrc/contributing_docs.rst) for contributor
commands and publishing instructions. Run setup commands from the repository
root in a Conda-initialized terminal.
Let the user create or change environments unless execution is requested.

## Environment

- `environment.yml` supplies the runtime/test dependencies.
- `environment-doc.yml` adds documentation tools and Sphinx `>=8.2,<9`.
- Create `vtools3_doc` from the runtime YAML, then update it with the overlay
  without `--prune`. The overlay is not a standalone environment.
- `build_doc_env.bat` and `build_doc_env.sh` are convenience wrappers over those
  steps and the editable installation, not alternative dependency definitions.
- Activate `vtools3_doc` before building. Do not modify another existing
  environment merely to make this documentation build work.
- Install the checkout with `python -m pip install --no-deps -e .`.
  Adding `[doc]` with `--no-deps` does not install extra dependencies.
- An existing environment is not disposable. Update a compatible environment,
  or ask before replacing it.

## Build

From `docsrc`, run `make html` (`.\make.bat html` in Windows PowerShell).
For a fresh build, run `make clean` and then `make html`.
`doall.bat` is a compatibility alias, not a separate API-generation procedure.

`sphinx.ext.apidoc` generates API RST under ignored `docsrc/_api/` during every
normal build. Do not edit or commit generated RST/HTML, or run a separate
`sphinx-apidoc` command into `docsrc`.

HTML goes to ignored `docs/html`; inspect `index.html`, affected API sections
and the notebook pages rather than treating successful HTML generation as
completeness.

## Failures and warnings

- `PandocMissing` from nbsphinx usually means `vtools3_doc` is not activated
  (pandoc is installed in the environment, not system-wide).
- Missing API content: examine import warnings in the docs environment.
  Correct dependency inputs or report a source-level failure; do not add mock
  imports or suppress warnings merely to produce a green build.
- Existing docstring and notebook warnings remain (about 130 at the time of
  writing). Warning-free builds are not currently required; do not hide those
  warnings or introduce unrelated fixes.

## CI and deployment

The docs workflow uses the same runtime YAML, overlay and checkout installation,
then `make html`. Do not install a released vtools3 over the checkout or restore
a separate apidoc step.

GitHub Pages receives `docs/html` through upload/deploy actions, not a branch.
Settings -> Pages must use GitHub Actions, and the `github-pages` environment
must allow the publishing branch/tag. Pull-request checks and branch protection
are outside this change.
