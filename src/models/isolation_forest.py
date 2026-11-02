"""Isolation Forest implemented from scratch with NumPy.

Everyday Analogy:
-----------------
Imagine you are playing '20 Questions' to find a specific person in a crowded train station.
Suppose there is one passenger dressed in a full neon-green astronaut spacesuit.
How many questions would it take to isolate them? Just one:
"Are you wearing a bright neon astronaut suit?" -> Yes! (Isolated in 1 question).
Now imagine trying to isolate an ordinary passenger wearing blue jeans and a black jacket.
You would have to ask many questions: "Are you wearing jeans?", "Do you have glasses?",
"Are you holding an umbrella?", "What color is your backpack?"...
Anomalies are few and distinctly different, so random, recursive questions isolate them
very quickly (short path lengths). Normal points are clustered densely together, requiring
many cuts (long path lengths) to isolate.

Algorithm Description:
----------------------
1. Forest Construction:
   - For each tree (e.g., 50 or 100 trees), sample 256 transactions without replacement.
   - Max height limit = ceil(log2(256)) = 8.
   - At each node, choose a random feature j and a random split threshold uniformly
     between min(X[:, j]) and max(X[:, j]).
   - Partition data into left (< threshold) and right (>= threshold).
   - Recurse until max depth is reached, sample size <= 1, or all values are identical.
   - Trees are stored in flattened NumPy arrays for fully vectorized traversal.

2. Path Length & Anomaly Score:
   - For any transaction x, traverse each tree to determine its path length h(x).
   - If traversal ends at a leaf with size s > 1, add Euler constant adjustment c(s).
   - Compute average path length E[h(x)] across all trees.
   - Anomaly score = 2^(-E[h(x)] / c(subsample)), where c(n) is the average path length
     of an unsuccessful search in a binary search tree.
   - Scores approach 1 for anomalies (short paths) and approach 0 for normal transactions.
"""

import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


def c_factor(n: int) -> float:
    """Average path length of unsuccessful search in a Binary Search Tree (BST).

    Formula (Liu et al., 2008):
        c(n) = 2 * (ln(n - 1) + 0.5772156649) - (2 * (n - 1) / n)  for n > 2
        c(2) = 1.0
        c(n) = 0.0  for n <= 1

    Euler-Mascheroni constant: ~0.5772156649
    """
    if n > 2:
        euler_mascheroni = 0.5772156649
        return float(2.0 * (np.log(n - 1) + euler_mascheroni) - (2.0 * (n - 1) / n))
    elif n == 2:
        return 1.0
    else:
        return 0.0


class IsolationTree:
    """A single isolation tree represented as flat parallel NumPy arrays for vectorized traversal."""

    def __init__(self):
        self.feature: np.ndarray = np.empty(0, dtype=np.int16)
        self.threshold: np.ndarray = np.empty(0, dtype=np.float32)
        self.left_child: np.ndarray = np.empty(0, dtype=np.int32)
        self.right_child: np.ndarray = np.empty(0, dtype=np.int32)
        self.node_size: np.ndarray = np.empty(0, dtype=np.int32)

    def build(self, X_sub: np.ndarray, max_height: int, rng: np.random.Generator):
        """Recursively construct tree and serialize into flat parallel arrays."""
        features_list: List[int] = []
        thresholds_list: List[float] = []
        left_list: List[int] = []
        right_list: List[int] = []
        size_list: List[int] = []

        def _build_node(indices: np.ndarray, current_height: int) -> int:
            node_idx = len(features_list)
            n_samples = len(indices)

            # Base cases: maximum depth reached or insufficient samples to split
            if current_height >= max_height or n_samples <= 1:
                features_list.append(-1)
                thresholds_list.append(0.0)
                left_list.append(-1)
                right_list.append(-1)
                size_list.append(n_samples)
                return node_idx

            X_node = X_sub[indices]
            n_features = X_node.shape[1]

            # Try to find a valid feature with min != max
            feat_order = rng.permutation(n_features)
            chosen_feat = -1
            min_val = 0.0
            max_val = 0.0

            for f in feat_order:
                col = X_node[:, f]
                col_min = float(np.min(col))
                col_max = float(np.max(col))
                if col_min < col_max:
                    chosen_feat = f
                    min_val = col_min
                    max_val = col_max
                    break

            # If all features are constant, make this a leaf
            if chosen_feat == -1:
                features_list.append(-1)
                thresholds_list.append(0.0)
                left_list.append(-1)
                right_list.append(-1)
                size_list.append(n_samples)
                return node_idx

            # Uniform random threshold between min and max
            threshold = float(rng.uniform(min_val, max_val))

            # Split indices
            col_vals = X_node[:, chosen_feat]
            left_mask = col_vals < threshold
            left_indices = indices[left_mask]
            right_indices = indices[~left_mask]

            # Reserve node slot in lists
            features_list.append(chosen_feat)
            thresholds_list.append(threshold)
            left_list.append(-1)   # Placeholder to be filled
            right_list.append(-1)  # Placeholder to be filled
            size_list.append(n_samples)

            # Recurse children
            left_child_idx = _build_node(left_indices, current_height + 1)
            right_child_idx = _build_node(right_indices, current_height + 1)

            left_list[node_idx] = left_child_idx
            right_list[node_idx] = right_child_idx

            return node_idx

        all_indices = np.arange(len(X_sub))
        _build_node(all_indices, current_height=0)

        self.feature = np.array(features_list, dtype=np.int16)
        self.threshold = np.array(thresholds_list, dtype=np.float32)
        self.left_child = np.array(left_list, dtype=np.int32)
        self.right_child = np.array(right_list, dtype=np.int32)
        self.node_size = np.array(size_list, dtype=np.int32)

    def compute_path_lengths(self, X: np.ndarray, max_height: int) -> np.ndarray:
        """Vectorized traversal of all rows in X through this tree."""
        n_samples = len(X)
        curr_nodes = np.zeros(n_samples, dtype=np.int32)
        depths = np.zeros(n_samples, dtype=np.float32)
        active = np.ones(n_samples, dtype=bool)

        for _ in range(max_height + 1):
            if not np.any(active):
                break

            active_indices = np.where(active)[0]
            nodes = curr_nodes[active_indices]
            feats = self.feature[nodes]

            # Detect leaf nodes (feat == -1)
            is_leaf = feats == -1
            if np.any(is_leaf):
                leaf_sample_idx = active_indices[is_leaf]
                leaf_nodes = nodes[is_leaf]
                leaf_sizes = self.node_size[leaf_nodes]

                # Add Euler adjustment for unexpanded leaf size
                adjustments = np.array([c_factor(int(s)) for s in leaf_sizes], dtype=np.float32)
                depths[leaf_sample_idx] += adjustments
                active[leaf_sample_idx] = False

            # Traverse internal nodes
            is_internal = ~is_leaf
            if np.any(is_internal):
                internal_sample_idx = active_indices[is_internal]
                internal_nodes = nodes[is_internal]
                split_feats = feats[is_internal]
                split_thresh = self.threshold[internal_nodes]

                # Check if features are less than threshold
                vals = X[internal_sample_idx, split_feats]
                go_left = vals < split_thresh

                # Advance node pointers
                curr_nodes[internal_sample_idx] = np.where(
                    go_left,
                    self.left_child[internal_nodes],
                    self.right_child[internal_nodes],
                )
                depths[internal_sample_idx] += 1.0

        return depths

    def to_dict(self) -> Dict[str, Any]:
        """Convert tree arrays to a serializable dictionary."""
        return {
            "feature": self.feature.tolist(),
            "threshold": self.threshold.tolist(),
            "left_child": self.left_child.tolist(),
            "right_child": self.right_child.tolist(),
            "node_size": self.node_size.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IsolationTree":
        """Reconstruct tree from dictionary."""
        t = cls()
        t.feature = np.array(d["feature"], dtype=np.int16)
        t.threshold = np.array(d["threshold"], dtype=np.float32)
        t.left_child = np.array(d["left_child"], dtype=np.int32)
        t.right_child = np.array(d["right_child"], dtype=np.int32)
        t.node_size = np.array(d["node_size"], dtype=np.int32)
        return t


class IsolationForestFromScratch:
    """Ensemble of isolation trees implemented completely from scratch.

    Parameters
    ----------
    n_trees : int, default=100
        Number of trees in the forest. Common values: 50, 100.
    subsample_size : int, default=256
        Number of samples drawn to build each tree.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_trees: int = 100,
        subsample_size: int = 256,
        random_state: int = 42,
    ):
        if n_trees <= 0:
            raise ValueError(f"n_trees must be positive, got {n_trees}")
        if subsample_size <= 1:
            raise ValueError(f"subsample_size must be > 1, got {subsample_size}")

        self.n_trees = int(n_trees)
        self.subsample_size = int(subsample_size)
        self.random_state = int(random_state)
        # Height limit = ceil(log2(subsample_size))
        self.max_height = int(math.ceil(math.log2(self.subsample_size)))
        self.c_subsample = c_factor(self.subsample_size)

        self.trees: List[IsolationTree] = []
        self.is_fitted: bool = False

    def fit(self, X_normal: np.ndarray) -> "IsolationForestFromScratch":
        """Build ensemble of isolation trees from normal transactions.

        Parameters
        ----------
        X_normal : np.ndarray
            2D array of normal transactions, shape (n_samples, n_features).

        Returns
        -------
        self : IsolationForestFromScratch
            Fitted forest.
        """
        X = np.asarray(X_normal, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")

        n_samples = len(X)
        sample_size = min(self.subsample_size, n_samples)
        self.c_subsample = c_factor(sample_size)

        rng = np.random.default_rng(self.random_state)
        self.trees = []

        for _ in range(self.n_trees):
            # Draw subsample without replacement
            subsample_idx = rng.choice(n_samples, size=sample_size, replace=False)
            X_sub = X[subsample_idx]

            tree = IsolationTree()
            tree.build(X_sub, max_height=self.max_height, rng=rng)
            self.trees.append(tree)

        self.is_fitted = True
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly score for each transaction.

        Score formula:
            score(x) = 2^(-E[h(x)] / c(subsample))
        where E[h(x)] is the average path length across all trees.

        Higher score = shorter path length = isolated faster = more suspicious.
        Scores fall in (0, 1].

        Parameters
        ----------
        X : np.ndarray
            2D array of transactions to score, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            1D float32 array of anomaly scores, shape (n_samples,).
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        n_samples = len(X_arr)
        total_path_lengths = np.zeros(n_samples, dtype=np.float64)

        # Vectorized scoring across each tree
        for tree in self.trees:
            paths = tree.compute_path_lengths(X_arr, max_height=self.max_height)
            total_path_lengths += paths

        avg_path_length = total_path_lengths / self.n_trees

        # Isolation score: 2^(-E[h] / c(n))
        if self.c_subsample > 0:
            scores = 2.0 ** (-avg_path_length / self.c_subsample)
        else:
            scores = np.ones(n_samples, dtype=np.float32) * 0.5

        return scores.astype(np.float32)

    def to_state(self) -> Dict[str, Any]:
        """Serialize forest state to a dictionary without pickle."""
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize an unfitted IsolationForestFromScratch.")

        return {
            "model_type": "IsolationForestFromScratch",
            "n_trees": self.n_trees,
            "subsample_size": self.subsample_size,
            "random_state": self.random_state,
            "max_height": self.max_height,
            "c_subsample": self.c_subsample,
            "trees": [tree.to_dict() for tree in self.trees],
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "IsolationForestFromScratch":
        """Reconstruct forest from serialized state dictionary."""
        instance = cls(
            n_trees=state["n_trees"],
            subsample_size=state["subsample_size"],
            random_state=state["random_state"],
        )
        instance.max_height = state["max_height"]
        instance.c_subsample = state["c_subsample"]
        instance.trees = [IsolationTree.from_dict(t) for t in state["trees"]]
        instance.is_fitted = bool(state.get("is_fitted", True))
        return instance
