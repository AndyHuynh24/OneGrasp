import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from touch.train import SHM_TIMEOUT, _tolerant


class Flaky(Dataset):
    def __init__(self, bad, msg=SHM_TIMEOUT):
        self.bad, self.msg = bad, msg

    def __len__(self):
        return 6

    def __getitem__(self, i):
        if i in self.bad:
            raise RuntimeError(self.msg)
        return torch.tensor(i)


def test_tolerant_skips_shm_timeouts_only():
    got = [int(b) for b in _tolerant(DataLoader(Flaky({1, 4}), batch_size=1))]
    assert got == [0, 2, 3, 5]
    with pytest.raises(RuntimeError, match="boom"):
        list(_tolerant(DataLoader(Flaky({2}, "boom"), batch_size=1)))
    with pytest.raises(RuntimeError, match="timed out"):
        list(_tolerant(DataLoader(Flaky({1, 2, 3}), batch_size=1), max_skips=2))
