"""REPORT.md, the committed output of the full run, against the paper's findings."""

from __future__ import annotations

import re

import pytest


def rank1(report: str, model: str, condition: str) -> float:
    match = re.search(rf"\| {model} \| {condition} \| ([\d.]+)% \|", report)
    assert match, f"{model}/{condition} missing from REPORT.md"
    return float(match.group(1))


def verification(report: str, model: str, protocol: str) -> float:
    match = re.search(rf"\| {model} \| {protocol} \| ([\d.]+)% \| ([\d.]+)% \|", report)
    assert match, f"{model}/{protocol} missing from REPORT.md"
    return float(match.group(1))


def test_identification_works_on_clean_probes(report, targets):
    """Sec. III: the matchers are near perfect on mugshot-quality probes."""
    assert rank1(report, "clean", "probe_clean") >= targets["clean_rank1_floor"]


def test_verification_works_on_clean_pairs(report, targets):
    assert verification(report, "clean", "lfw_none") >= targets["clean_lfw_accuracy_floor"]


def test_sunglasses_on_the_probe_reduce_identification_accuracy(report):
    """Abstract: sunglasses in the probe image substantially reduce one-to-many accuracy."""
    assert rank1(report, "clean", "probe_sunglasses") < rank1(report, "clean", "probe_clean")


@pytest.mark.parametrize("dataset", ["lfw", "calfw", "cplfw"])
def test_sunglasses_on_the_probe_reduce_verification_accuracy(report, dataset):
    assert verification(report, "clean", f"{dataset}_probe") < verification(report, "clean", f"{dataset}_none")


def test_adding_sunglasses_to_the_gallery_recovers_accuracy(report):
    """Table V: adding sunglasses to all gallery images recovers part of the lost accuracy."""
    assert rank1(report, "clean", "gallery_sunglasses_probe_sunglasses") > rank1(
        report, "clean", "probe_sunglasses"
    )


def test_training_with_sunglasses_helps_on_sunglasses_probes(report):
    """Sec. IV-C, Fig. 9: more sunglasses images in training lowers the error on sunglasses probes."""
    assert rank1(report, "sunglass_aware", "probe_sunglasses") > rank1(report, "clean", "probe_sunglasses")


def test_training_with_sunglasses_keeps_general_accuracy(report, targets):
    """Table VI: general accuracy is unchanged by adding sunglasses images to training."""
    assert verification(report, "sunglass_aware", "lfw_none") == pytest.approx(
        verification(report, "clean", "lfw_none"), abs=targets["general_accuracy_tolerance_pp"]
    )
