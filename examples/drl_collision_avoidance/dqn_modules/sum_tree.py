import numpy as np


class SumTree:
    """Optimized SumTree for Prioritized Experience Replay (priority-only).

    The tree is a flat array in heap order: node 0 is the root, node i's
    children are 2i+1 and 2i+2, and leaves start at index ``capacity - 1``.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data_pointer = 0
        self.max_priority = 1.0
        # Precompute tree depth for vectorized descent/propagation.
        self._depth = int(np.ceil(np.log2(max(capacity, 1))))
        if (1 << self._depth) < capacity:
            self._depth += 1

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

    # ---- Vectorized batch operations ----

    def sample_batch(self, bs: int):
        """Vectorized level-by-level descent for ``bs`` samples at once.

        Returns (tree_indices, priorities, data_indices) arrays of shape (bs,).
        Replaces ``bs`` individual ``get()`` calls (each walking ~20 levels
        in a Python while loop) with ~20 vectorized numpy operations.
        """
        total = float(self.tree[0])
        if total <= 0.0 or not np.isfinite(total):
            # Degenerate: uniform random
            data_idx = np.random.randint(0, max(self.capacity, 1), size=bs)
            tree_idx = data_idx + self.capacity - 1
            return tree_idx, self.tree[tree_idx], data_idx

        segment = total / bs
        # Stratified sampling: one random value per segment
        s = np.random.uniform(
            segment * np.arange(bs, dtype=np.float32),
            segment * (np.arange(1, bs + 1, dtype=np.float32)),
        )
        s = np.clip(s, 0.0, np.nextafter(total, 0.0))

        indices = np.zeros(bs, dtype=np.int64)
        for _ in range(self._depth):
            left = 2 * indices + 1
            right = left + 1
            # Guard against going past the tree (for non-power-of-2 capacity)
            in_tree = left < len(self.tree)
            left_safe = np.where(in_tree, left, 0)
            right_safe = np.where(in_tree, right, 0)
            left_sum = self.tree[left_safe]
            go_left = (s <= left_sum) & in_tree
            indices = np.where(go_left, left_safe, right_safe)
            s = np.where(go_left, s, s - left_sum)

        # Clamp to valid leaf range
        indices = np.clip(indices, self.capacity - 1, 2 * self.capacity - 2)
        data_idx = indices - self.capacity + 1
        return indices, self.tree[indices], data_idx

    def update_batch(self, tree_indices: np.ndarray, priorities: np.ndarray):
        """Vectorized level-by-level propagation for ``bs`` updates at once.

        Replaces ``bs`` individual ``update()`` calls (each walking ~20
        levels up in a Python while loop) with ~20 vectorized numpy
        scatter-add operations.
        """
        priorities = np.asarray(priorities, dtype=np.float32)
        tree_indices = np.asarray(tree_indices, dtype=np.int64)
        # Sanitize
        valid = np.isfinite(priorities) & (priorities > 0.0)
        priorities = np.where(valid, priorities, 1e-6)

        old = self.tree[tree_indices]
        self.tree[tree_indices] = priorities
        changes = priorities - old

        # Update max_priority
        self.max_priority = max(self.max_priority, float(priorities.max()))

        # Propagate changes up the tree, level by level.
        indices = tree_indices.copy()
        deltas = changes.copy()
        for _ in range(self._depth):
            parents = (indices - 1) // 2
            # Scatter-add deltas to parents (handles duplicate parents where
            # two children share the same parent). Use sort-based reduction
            # to avoid the slow np.add.at unbuffered-ufunc path.
            if len(parents) == 0:
                break
            order = np.argsort(parents, kind='stable')
            sp = parents[order]
            sd = deltas[order]
            starts = np.concatenate(([0], np.flatnonzero(np.diff(sp)) + 1))
            sums = np.add.reduceat(sd, starts)
            unique_parents = sp[starts]
            self.tree[unique_parents] += sums
            indices = unique_parents
            deltas = sums
