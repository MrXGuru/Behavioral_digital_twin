"""Unit tests for the categorical encoders.

Covers Requirements 4.5 and 4.6:

* 4.5 -- encoders apply the encoding learned during ``fit`` and produce fixed,
  schema-versioned dimensions (stable output width regardless of transform-time
  values).
* 4.6 -- categorical values unseen during ``fit`` map to the explicit ``<UNK>``
  bucket without raising an error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from data.schema import UNK
from features.encoders import ContextEncoder, OneHotEncoder


# ---------------------------------------------------------------------------
# OneHotEncoder
# ---------------------------------------------------------------------------


def test_seen_value_produces_correct_one_hot():
    """A seen value yields a single 1.0 at its sorted-vocabulary index."""
    enc = OneHotEncoder().fit(["work", "home", "gym"])
    # Vocabulary is sorted: ["gym", "home", "work"].
    assert enc.vocabulary == ["gym", "home", "work"]

    vec = enc.transform_one("home")
    assert vec == [0.0, 1.0, 0.0, 0.0]  # index 1 hot, trailing <UNK> slot 0.
    assert sum(vec) == 1.0
    assert vec[enc.index_of("home")] == 1.0


def test_unseen_value_maps_to_unk_without_raising():
    """An unseen value maps to the <UNK> slot and does not raise (Req 4.6)."""
    enc = OneHotEncoder().fit(["home", "work"])

    # Must not raise on an unseen category.
    vec = enc.transform_one("space_station")

    assert enc.index_of("space_station") == enc.unk_index
    assert vec[enc.unk_index] == 1.0
    assert sum(vec) == 1.0  # exactly the UNK slot is hot.


def test_output_dim_is_vocab_plus_unk_and_stable(recwarn):
    """Width == len(vocabulary) + 1 and is stable across transform values (Req 4.5)."""
    enc = OneHotEncoder().fit(["a", "b", "c"])
    assert enc.output_dim == len(enc.vocabulary) + 1 == 4
    assert enc.width == enc.output_dim

    # Transforming seen, unseen, and mixed values never changes the width.
    assert len(enc.transform_one("a")) == 4
    assert len(enc.transform_one("zzz_unseen")) == 4
    matrix = enc.transform(["a", "b", "unseen", "c"])
    assert matrix.shape == (4, 4)


def test_fit_transform_consistency():
    """The same value always maps to the same index across calls (Req 4.5)."""
    enc = OneHotEncoder().fit(["x", "y", "z"])
    first = enc.index_of("y")
    for _ in range(5):
        assert enc.index_of("y") == first
        assert enc.transform_one("y") == enc.transform_one("y")


def test_transform_matches_transform_one():
    """Batch transform rows equal the per-value transform_one output."""
    enc = OneHotEncoder().fit(["red", "green", "blue"])
    values = ["red", "blue", "unseen", "green"]
    matrix = enc.transform(values)
    for i, value in enumerate(values):
        assert matrix[i].tolist() == enc.transform_one(value)


def test_one_hot_feature_names_length_matches_output_dim():
    """feature_names length equals output_dim and ends with the <UNK> column."""
    enc = OneHotEncoder().fit(["home", "work"])
    names = enc.feature_names("location")
    assert len(names) == enc.output_dim
    assert names[-1] == f"location={UNK}"


# ---------------------------------------------------------------------------
# ContextEncoder
# ---------------------------------------------------------------------------


@dataclass
class _Row:
    """Minimal DecisionRecord-like object exposing attributes used by the encoder."""

    location: str
    weather: str


def _dict_rows():
    return [
        {"location": "home", "weather": "clear"},
        {"location": "work", "weather": "rain"},
    ]


def test_context_output_dim_is_sum_of_column_widths():
    """output_dim equals the sum of per-column widths (Req 4.5)."""
    enc = ContextEncoder().fit(_dict_rows())
    widths = enc.column_widths()
    assert enc.output_dim == sum(widths.values())
    # 2 locations + UNK == 3, 2 weathers + UNK == 3 -> 6 total.
    assert widths == {"location": 3, "weather": 3}
    assert enc.output_dim == 6


def test_context_transform_one_handles_dict_and_object():
    """transform_one works on both dict rows and attribute objects, equally."""
    enc = ContextEncoder().fit(_dict_rows())

    dict_vec = enc.transform_one({"location": "home", "weather": "rain"})
    obj_vec = enc.transform_one(_Row(location="home", weather="rain"))

    assert isinstance(dict_vec, np.ndarray)
    assert dict_vec.shape == (enc.output_dim,)
    np.testing.assert_array_equal(dict_vec, obj_vec)


def test_context_unseen_values_fall_through_to_column_unk():
    """Unseen values in any column map to that column's UNK without raising (Req 4.6)."""
    enc = ContextEncoder().fit(_dict_rows())

    vec = enc.transform_one({"location": "moon_base", "weather": "snow"})

    loc_enc = enc.encoder("location")
    wx_enc = enc.encoder("weather")
    # location sub-vector occupies the first loc_enc.width slots.
    loc_part = vec[: loc_enc.width]
    wx_part = vec[loc_enc.width : loc_enc.width + wx_enc.width]

    assert loc_part[loc_enc.unk_index] == 1.0
    assert wx_part[wx_enc.unk_index] == 1.0
    assert loc_part.sum() == 1.0
    assert wx_part.sum() == 1.0


def test_context_output_dim_stable_regardless_of_transform_values():
    """Width stays fixed whether values are seen, unseen, or mixed (Req 4.5)."""
    enc = ContextEncoder().fit(_dict_rows())
    expected = enc.output_dim

    assert enc.transform_one({"location": "home", "weather": "clear"}).shape == (expected,)
    assert enc.transform_one({"location": "?", "weather": "?"}).shape == (expected,)

    matrix = enc.transform(
        [
            {"location": "home", "weather": "clear"},
            {"location": "unseen", "weather": "rain"},
        ]
    )
    assert matrix.shape == (2, expected)


def test_context_feature_names_length_matches_output_dim():
    """feature_names length equals output_dim (Req 4.5)."""
    enc = ContextEncoder().fit(_dict_rows())
    names = enc.feature_names()
    assert len(names) == enc.output_dim
    assert f"location={UNK}" in names
    assert f"weather={UNK}" in names
