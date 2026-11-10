"""Synthetic sunglasses and the verification error rate, on synthetic inputs."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sungfr.augment.sunglasses import apply_sunglasses
from sungfr.eval import best_acc_eer


def test_sunglasses_cover_only_the_eye_region():
    """Sec. III: sunglasses occlude the eyes and leave the rest of the face as it was."""
    x = torch.rand(2, 3, 112, 112) * 2 - 1
    out = apply_sunglasses(x)
    eyes = slice(int(0.28 * 112), int(0.48 * 112))
    assert torch.equal(out[:, :, : eyes.start], x[:, :, : eyes.start])
    assert torch.equal(out[:, :, eyes.stop :], x[:, :, eyes.stop :])
    assert (out[:, :, eyes] != x[:, :, eyes]).float().mean() > 0.3


def test_lenses_are_dark():
    out = apply_sunglasses(torch.ones(1, 3, 112, 112))
    lens = out[0, :, int(0.38 * 112), int(0.32 * 112)]
    assert (lens < -0.8).all()


def test_single_images_are_supported():
    x = torch.rand(3, 112, 112)
    assert apply_sunglasses(x).shape == x.shape


def test_eer_is_where_false_match_and_false_non_match_rates_meet():
    """The EER is the error rate where FMR equals FNMR, not the lowest half total error rate."""
    scores = np.array([0.6, 0.7, 0.8, 0.9, 0.0, 0.1, 0.2, 0.65])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert best_acc_eer(scores, labels)[1] == pytest.approx(0.25, abs=0.01)


def test_perfectly_separated_scores_have_no_errors():
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([1, 1, 0, 0])
    acc, eer = best_acc_eer(scores, labels)[:2]
    assert acc == pytest.approx(1.0)
    assert eer == pytest.approx(0.0)
