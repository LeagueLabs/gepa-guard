import inspect
from importlib import metadata
from importlib.resources import files

import pytest

import gepa_guard


class TestPublicApi:
    def test_version_matches_distribution(self) -> None:
        assert gepa_guard.__version__ == metadata.version("gepa-guard")

    def test_exports_are_stable(self) -> None:
        assert sorted(gepa_guard.__all__) == [
            "BaseCriterion",
            "Evaluator",
            "FloorConfig",
            "FloorRegression",
            "GuardedAcceptance",
            "ProtectedFloor",
            "Regression",
            "UnguardedMergeError",
            "VetoStats",
        ]

    def test_all_names_resolve(self) -> None:
        assert all(hasattr(gepa_guard, name) for name in gepa_guard.__all__)

    @pytest.mark.parametrize(
        ("owner", "parameter", "expected"),
        [
            ("GuardedAcceptance", "base", "BaseCriterion"),
            ("ProtectedFloor", "evaluate", "Evaluator"),
        ],
    )
    def test_extension_points_have_public_types(
        self, owner: str, parameter: str, expected: str
    ) -> None:
        signature = inspect.signature(getattr(gepa_guard, owner))
        assert signature.parameters[parameter].annotation is getattr(
            gepa_guard, expected
        )

    def test_package_is_typed(self) -> None:
        assert files("gepa_guard").joinpath("py.typed").is_file()
