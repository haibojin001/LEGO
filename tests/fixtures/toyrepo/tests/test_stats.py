import pytest
from toymath import mean, total


def test_total():
    assert total([1, 2, 3]) == 6


def test_total_empty():
    assert total([]) == 0


def test_mean():
    assert mean([2, 4]) == 3


def test_mean_empty():
    with pytest.raises(ValueError):
        mean([])


def test_version_free():
    assert 1 + 1 == 2
