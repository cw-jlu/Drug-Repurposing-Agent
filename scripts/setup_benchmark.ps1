# Run from repository root with Python 3.10 on Windows.
$ErrorActionPreference = 'Stop'
python -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install pip==26.2.1 setuptools==84.0.0
& $python -m pip install -r requirements-benchmark.lock --no-deps
# Legacy stanscofi/benchscofi dependencies declare conflicting NumPy pins.
# The lock captures the tested NumPy 1.26.4 runtime, so install without
# dependency re-resolution and then add this checkout as a local package.
& $python -m pip install -e . --no-deps
& $python -m pytest -q
