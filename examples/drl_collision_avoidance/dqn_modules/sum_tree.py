import numpy as np

class SumTree:
    """Optimized SumTree for Prioritized Experience Replay (priority-only)."""
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data_pointer = 0
        self.max_priority = 1.0

    def add(self, priority: float):
        """Add new experience with given priority (new ones get max priority)."""
        priority = float(priority)
        if not np.isfinite(priority) or priority <= 0.0:
            priority = 1e-6
        index = self.data_pointer + self.capacity - 1
        self.update(index, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, index: int, priority: float):
        """Update leaf and propagate change up the tree."""
        priority = float(priority)
        if not np.isfinite(priority) or priority <= 0.0:
            priority = 1e-6
        change = priority - self.tree[index]
        self.tree[index] = priority
        self._propagate(index, change)
        self.max_priority = max(self.max_priority, priority)

    def _propagate(self, index: int, change: float):
        """Iterative propagation (fast & safe)."""
        while index > 0:
            index = (index - 1) // 2
            self.tree[index] += change

    def _retrieve(self, index: int, s: float):
        """Find leaf for cumulative sum s."""
        while True:
            left = 2 * index + 1
            if left >= len(self.tree):  # leaf node
                return index
            if s <= self.tree[left]:
                index = left
            else:
                s -= self.tree[left]
                index = left + 1

    def get(self, s: float):
        """Get tree index, priority, and data index."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], data_idx  # tree_idx, priority, data_idx

    def total(self) -> float:
        return float(self.tree[0])
