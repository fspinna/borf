from pathlib import Path

import fast_borf

REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config):
    # fast_borf can also be installed in editable mode from another checkout,
    # so make sure the tests exercise the code in this one.
    package_path = Path(fast_borf.__file__).resolve()
    if not package_path.is_relative_to(REPO_ROOT):
        raise RuntimeError(
            f"fast_borf was imported from {package_path}, expected it under {REPO_ROOT}"
        )
