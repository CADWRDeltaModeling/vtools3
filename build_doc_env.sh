#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
    -h|--help)
        printf '%s\n' \
            'Usage: bash build_doc_env.sh' \
            'Create vtools3_doc from environment.yml plus environment-doc.yml,' \
            'then install this checkout editable. Requires Conda on PATH.' \
            'An existing environment is not deleted or replaced.'
        exit 0
        ;;
    "")
        ;;
    *)
        printf '%s\n' 'Usage: bash build_doc_env.sh' >&2
        exit 2
        ;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
conda_hook="$(conda shell.bash hook)"
eval "$conda_hook"
conda env create -n vtools3_doc -f environment.yml
conda env update -n vtools3_doc -f environment-doc.yml
conda activate vtools3_doc
python -m pip install --no-deps -e .
printf '%s\n' \
    'Documentation environment ready.' \
    'Next: conda activate vtools3_doc' \
    'Then: cd docsrc' \
    '      make html'
