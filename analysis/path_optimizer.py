"""
Path optimizer — recommends optimized layer execution paths.

Uses compound analysis results and delta matrices to:
1. Recommend a layer execution path given domain weights
2. Identify which layers have the highest routing discriminative power
"""

import numpy as np

from sweep.runner import build_optimized_path


class PathOptimizer:
    """Optimizes layer execution paths based on sweep results and domain weights."""

    def __init__(self, compound_results: dict, n_layers: int):
        """
        Args:
            compound_results: Output from generate_compound_report().
                Must have keys: synergistic, antagonistic, cascade, inhibitory.
            n_layers: Total number of layers in the model.
        """
        self.compound_results = compound_results
        self.n_layers = n_layers

    def recommend_path(self, domain_weights: dict[str, float]) -> list[int]:
        """Given {probe_name: weight}, return optimized layer execution path.

        Algorithm:
        1. Get top circuit regions for each weighted probe from compound results
        2. Weight recommendations by probe weight
        3. Resolve conflicts (higher weight wins)
        4. Include synergistic circuits unconditionally
        5. Return combined path via build_optimized_path()

        Args:
            domain_weights: {probe_name: weight} where weight > 0 means
                the probe should be prioritized. Higher weight = more priority.

        Returns:
            Ordered list of layer indices representing the execution path.
        """
        if not domain_weights:
            return list(range(self.n_layers))

        # Collect per-layer scores: positive = duplicate, negative = skip
        layer_scores = np.zeros(self.n_layers)

        # Score from synergistic circuits (always beneficial)
        for circuit in self.compound_results.get("synergistic", []):
            ci, cj = circuit["i"], circuit["j"]
            # Check if any weighted probe is in the improving set
            relevant_weight = sum(
                domain_weights.get(p, 0.0)
                for p in circuit.get("improving_probes", [])
            )
            if relevant_weight > 0:
                for k in range(ci, min(cj, self.n_layers)):
                    layer_scores[k] += relevant_weight * circuit.get("mean_delta", 0.0)

        # Score from antagonistic circuits (use weighted preference)
        for circuit in self.compound_results.get("antagonistic", []):
            ci, cj = circuit["i"], circuit["j"]
            improved_weight = sum(
                domain_weights.get(p, 0.0)
                for p in circuit.get("improved", [])
            )
            degraded_weight = sum(
                domain_weights.get(p, 0.0)
                for p in circuit.get("degraded", [])
            )
            net = improved_weight - degraded_weight
            for k in range(ci, min(cj, self.n_layers)):
                layer_scores[k] += net * 0.5  # dampen antagonistic influence

        # Score from inhibitory circuits
        for circuit in self.compound_results.get("inhibitory", []):
            ci, cj = circuit["i"], circuit["j"]
            best_probe = circuit.get("best_probe", "")
            if best_probe in domain_weights and domain_weights[best_probe] > 0:
                w = domain_weights[best_probe]
                for k in range(ci, min(cj, self.n_layers)):
                    layer_scores[k] += w * circuit.get("best_delta", 0.0) * 0.3

        # Convert scores to skip/duplicate regions
        dup_threshold = 0.01
        skip_threshold = -0.01

        duplicate_regions = []
        skip_regions = []

        # Find contiguous runs
        k = 0
        while k < self.n_layers:
            if layer_scores[k] > dup_threshold:
                start = k
                while k < self.n_layers and layer_scores[k] > dup_threshold:
                    k += 1
                duplicate_regions.append((start, k))
            elif layer_scores[k] < skip_threshold:
                start = k
                while k < self.n_layers and layer_scores[k] < skip_threshold:
                    k += 1
                skip_regions.append((start, k))
            else:
                k += 1

        return build_optimized_path(self.n_layers, skip_regions, duplicate_regions)

    def recommend_router_features(
        self, matrices: dict[str, np.ndarray]
    ) -> dict[int, float]:
        """Identify which layer indices have highest predictive power for routing.

        For each layer, compute variance of deltas across probes when that layer
        is part of a duplicated block. High variance means the layer discriminates
        between domains and is therefore a good routing decision point.

        Args:
            matrices: {probe_name: NxM delta matrix} from build_delta_matrix.

        Returns:
            {layer_index: variance_score} sorted by score descending.
            Higher score = better routing feature.
        """
        if not matrices:
            return {}

        probe_names = list(matrices.keys())
        layer_variances = {}

        for layer in range(self.n_layers):
            # Collect all deltas for configs involving this layer
            per_probe_deltas = []
            for name in probe_names:
                mat = matrices[name]
                # Configs where this layer is in the duplicated range: i <= layer < j
                deltas = []
                for i in range(min(layer + 1, mat.shape[0])):
                    for j in range(layer + 1, mat.shape[1]):
                        val = mat[i, j]
                        if not np.isnan(val):
                            deltas.append(val)
                if deltas:
                    per_probe_deltas.append(float(np.mean(deltas)))

            if len(per_probe_deltas) >= 2:
                layer_variances[layer] = float(np.var(per_probe_deltas))
            else:
                layer_variances[layer] = 0.0

        # Sort by variance descending
        return dict(sorted(layer_variances.items(), key=lambda x: -x[1]))
