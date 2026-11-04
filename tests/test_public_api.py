from importlib import metadata
from importlib.resources import files

import gepa_guard


class TestPublicApi:
    def test_version_matches_distribution(self) -> None:
        assert gepa_guard.__version__ == metadata.version("gepa-guard")

    def test_all_names_resolve(self) -> None:
        assert all(hasattr(gepa_guard, name) for name in gepa_guard.__all__)

    def test_package_is_typed(self) -> None:
        assert files("gepa_guard").joinpath("py.typed").is_file()
