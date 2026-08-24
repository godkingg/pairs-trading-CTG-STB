"""Kiểm tra purged_kfold_splits() không rò rỉ dữ liệu giữa train/test."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import purged_kfold_splits


def test_no_overlap_between_train_and_test():
    for train_idx, test_idx in purged_kfold_splits(n_samples=498, n_splits=5, embargo=5):
        assert len(set(train_idx) & set(test_idx)) == 0


def test_embargo_actually_removes_neighbors():
    """Các điểm trong khoảng embargo quanh test set không được xuất hiện trong train."""
    n, embargo = 100, 5
    for train_idx, test_idx in purged_kfold_splits(n, n_splits=5, embargo=embargo):
        test_start, test_end = test_idx.min(), test_idx.max()
        near_boundary = set(range(max(0, test_start - embargo), test_start)) | \
                        set(range(test_end + 1, min(n, test_end + 1 + embargo)))
        assert near_boundary.isdisjoint(set(train_idx))


def test_all_indices_covered_across_test_folds():
    n = 500
    seen = set()
    for _, test_idx in purged_kfold_splits(n, n_splits=5, embargo=5):
        seen |= set(test_idx)
    assert seen == set(range(n))
