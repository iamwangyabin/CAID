from __future__ import annotations

from caidbench.data.scenario import ContinualScenario


def _scenario() -> ContinualScenario:
    scenario = ContinualScenario.__new__(ContinualScenario)
    scenario.transform_cfg = {
        "train": {"trsf": [{"_target_": "TrainTransform"}]},
        "test": {"trsf": [{"_target_": "TestTransform"}]},
    }
    scenario.test_pre_transform_cfg = {"trsf": [{"_target_": "TestPreTransform"}]}
    return scenario


def test_test_pre_transform_is_added_for_real_test_split() -> None:
    transform = _scenario()._transform_for_split("test", data_split="test")

    assert [step["_target_"] for step in transform["trsf"]] == [
        "TestPreTransform",
        "TestTransform",
    ]


def test_test_pre_transform_is_not_added_to_train_rows_using_test_transform() -> None:
    transform = _scenario()._transform_for_split("test", data_split="train")

    assert [step["_target_"] for step in transform["trsf"]] == ["TestTransform"]
