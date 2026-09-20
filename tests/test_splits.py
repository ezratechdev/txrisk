import numpy as np

from blockchain_analysis.evaluation.splits import random_split, temporal_split


def test_temporal_split_never_trains_on_the_future():
    t = np.array([1, 5, 34, 35, 49, 2])
    split = temporal_split(t, train_until=34)
    assert t[split.train].max() <= 34 < t[split.test].min()
    assert (split.train ^ split.test).all()


def test_random_split_is_a_partition_of_the_requested_size():
    split = random_split(1000, test_fraction=0.3, seed=1)
    assert split.test.sum() == 300
    assert (split.train ^ split.test).all()
