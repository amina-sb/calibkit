from importlib.metadata import version

import calibkit


def test_version_matches_metadata() -> None:
    assert calibkit.__version__ == version("calibkit")
