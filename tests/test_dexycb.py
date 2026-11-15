import numpy as np

from touch.data.dexycb import SPLIT_S1, SUBJECTS, labels_from_dist, split_of


def test_s1_split_disjoint_and_complete():
    ids = [i for v in SPLIT_S1.values() for i in v]
    assert sorted(ids) == list(range(10))
    assert split_of(SUBJECTS[6]) == "val" and split_of(SUBJECTS[7]) == "test"


def dist(mm):  # rows of [palm, thumb, index, middle, ring, pinky] in mm -> metres
    return np.asarray(mm, np.float64) / 1000.0


def test_labels_contact_ignore_band_and_hold():
    pd = dist([
        [50, 50, 50, 50, 50, 50],  # none
        [50, 2, 50, 50, 50, 50],  # thumb touches -> touch (not a grasp)
        [50, 2, 1, 50, 50, 50],  # thumb + index, object still -> touch
        [50, 2, 1, 50, 50, 50],  # same, object lifted -> hold
        [50, 50, 6, 50, 50, 50],  # index in the 3..10 mm band only -> ignore
        [2, 50, 50, 50, 50, 50],  # palm only -> touch, fingers clear
    ])  # fmt: skip
    disp = np.array([0, 0, 0, 0.05, 0, 0])
    valid = np.ones(6, bool)
    L = labels_from_dist(pd, disp, valid)
    assert L["state"].tolist() == [0, 1, 1, 2, -1, 1]
    assert L["finger"][1].tolist() == [1, 0, 0, 0, 0]
    assert L["finger"][4].tolist() == [0, -1, 0, 0, 0]
    assert L["finger"][5].tolist() == [0, 0, 0, 0, 0]


def test_labels_invalid_frames_ignored():
    pd = dist([[1, 1, 1, 1, 1, 1]] * 2)
    L = labels_from_dist(pd, np.array([0.1, 0.1]), np.array([True, False]))
    assert L["state"].tolist() == [2, -1]
    assert (L["finger"][1] == -1).all()
