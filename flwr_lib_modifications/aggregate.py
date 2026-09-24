''' PATH: .venv/lib/python3.10/site-packages/flwr/server/strategy/aggregate.py'''

# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Aggregation functions vendored from `flwr.server.strategy.aggregate`,
extended with GLow's trust-weighted strategies, dispatched by
GLow_strategy.aggregate_fit() via `self.aggregation`:

- `aggregate_inplace` ('inplace'): plain sample-count-weighted average.
- `aggregate_score`/`aggregate_score_validation`: weight by each
  neighbour's own accuracy (centralized / self-reported).
- `aggregate_score_centroids_2` ('approach_2'): per-class distance/
  confidence/size score.

`results` is `{topology_index: FitRes}` -- built via `_cid_to_index` since
ClientProxy.cid is an opaque simulation id, not the topology index.

Flower's original generic/Byzantine-robust aggregation rules (Krum, Bulyan,
trimmed-mean, Q-FFL, plain median/weighted-average) are kept below the GLow
ones -- none of them are wired into this dispatch.
"""

from functools import reduce
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import json

from flwr.common import FitRes, NDArray, NDArrays, parameters_to_ndarrays

from scipy.spatial.distance import pdist, cdist, squareform, euclidean, cosine

def self_learning(results: Dict[int, FitRes], neighbours: List[int], head_id: int) -> NDArrays:
    """Self-Learning."""
    ordered_results = [(n, results[n]) for n in neighbours if n in results]
    params = []
    for idx, fit_res in ordered_results:
        if idx == head_id:
            params = parameters_to_ndarrays(fit_res.parameters)
    return params

def aggregate_inplace(results: Dict[int, FitRes]) -> NDArrays:
    """Compute in-place weighted average."""
    fit_results = [results[idx] for idx in sorted(results)]
    num_examples_total = sum(fit_res.num_examples for fit_res in fit_results)

    if num_examples_total > 0:
        scaling_factors = [
            fit_res.num_examples / num_examples_total for fit_res in fit_results
        ]
    else:
        scaling_factors = [
            1. / len(fit_results) for fit_res in fit_results
        ]

    params = [
        scaling_factors[0] * x for x in parameters_to_ndarrays(fit_results[0].parameters)
    ]
    for i, fit_res in enumerate(fit_results[1:]):
        res = (
            scaling_factors[i + 1] * x
            for x in parameters_to_ndarrays(fit_res.parameters)
        )
        params = [reduce(np.add, layer_updates) for layer_updates in zip(params, res)]
    return params


def aggregate_score(results: Dict[int, FitRes], neighbour_metrics: List[float], neighbours: List[int], head_id: int) -> NDArrays:
    """Weighted average where each neighbour's weight is its own centralized
    test-set accuracy from `neighbour_metrics` (the head instead uses its own
    just-reported `acc_val_distr`, since it has no `neighbour_metrics` entry
    for itself). Weights normalized to sum to 1."""

    ordered_results = [(n, results[n]) for n in neighbours if n in results]

    scaling_norm = 0.
    for idx, fit_res in ordered_results:
        if idx == head_id:
            scaling_norm += fit_res.metrics['acc_val_distr']
        else:
            if neighbour_metrics[idx] is not None:
                scaling_norm += neighbour_metrics[idx]

    if scaling_norm > 0:
        scaling_factors = []
        for idx, fit_res in ordered_results:
            if idx == head_id:
                scaling_factors.append(fit_res.metrics['acc_val_distr'] / scaling_norm)
            else:
                if neighbour_metrics[idx] is not None:
                    scaling_factors.append(neighbour_metrics[idx] / scaling_norm)
                else:
                    scaling_factors.append(0.)
    else:
        # no usable accuracy signal from anyone -- equal weighting instead
        # of silently zeroing out the aggregated model
        scaling_factors = [1. / len(ordered_results)] * len(ordered_results)

    params = [
        scaling_factors[0] * x for x in parameters_to_ndarrays(ordered_results[0][1].parameters)
    ]
    for i, (_, fit_res) in enumerate(ordered_results[1:]):
        res = (
            scaling_factors[i + 1] * x for x in parameters_to_ndarrays(fit_res.parameters)
        )
        params = [reduce(np.add, layer_updates) for layer_updates in zip(params, res)]
    return params

def aggregate_score_validation(results: Dict[int, FitRes], neighbours: List[int], head_id: int) -> NDArrays:
    """Like `aggregate_score`, but every participant (including the head) is
    weighted by its own self-reported `acc_val_distr` -- cheaper (no extra
    evaluation pass) but trusts each client's own report."""

    ordered_results = [(n, results[n]) for n in neighbours if n in results]

    scaling_norm = 0.
    for i, (idx, fit_res) in enumerate(ordered_results):
        scaling_norm += fit_res.metrics['acc_val_distr']

    if scaling_norm > 0:
        scaling_factors = [
            fit_res.metrics['acc_val_distr'] / scaling_norm for _, fit_res in ordered_results
        ]
    else:
        scaling_factors = [1. / len(ordered_results)] * len(ordered_results)

    params = [
        scaling_factors[0] * x for x in parameters_to_ndarrays(ordered_results[0][1].parameters)
    ]
    for i, (_, fit_res) in enumerate(ordered_results[1:]):
        res = (
            scaling_factors[i + 1] * x for x in parameters_to_ndarrays(fit_res.parameters)
        )
        params = [reduce(np.add, layer_updates) for layer_updates in zip(params, res)]
    return params

def aggregate_score_centroids_2(
    results: Dict[int, FitRes],
    neighbours: List[int],
    head_id: int,
    current_round: int,
    class_client_matrix: List[List[int]],
    class_number: int,
    alpha: float = 0.33,
    beta: float = 0.33,
    gamma: float = 0.33
) -> NDArrays:
    """approach_2: per-class score `distance**alpha * confidence**beta *
    size**gamma` per neighbour, normalized per class then summed and
    renormalized into one weight per neighbour. `distance` is Euclidean
    distance between head/neighbour mean predicted-probability vectors per
    class (`prob_matrix`)."""

    #log(INFO, "Strategy Aggregation for Head Client %s - Neighbours %s - (Round %s)", head_id, neighbours, current_round)
    eff_alpha, eff_beta, eff_gamma = alpha, beta, gamma

    ordered_results = [(n, results[n]) for n in neighbours if n in results]
    if head_id not in results:
        raise ValueError(f"Head {head_id}'s own fit() result is missing (Flower reported it as a failure) -- cannot aggregate without it.")
    present_neighbours = [n for n, _ in ordered_results]

    metrics_map = {idx: fit_res.metrics for idx, fit_res in ordered_results}

    effective_matrix = [list(row) for row in class_client_matrix]

    # true (pre-substitution) per-class coverage, for the DISTANCE gate below
    has_class = {n: [class_client_matrix[n][c] > 0 for c in range(class_number)] for n in present_neighbours}

    # CONFIDENCE -- crosses the wire JSON-encoded (legacy config bridge)
    v_conf = np.zeros((len(present_neighbours), class_number))
    head_confidence_data = json.loads(metrics_map[head_id].get('confidence_score', '[]'))

    for j, neighbour in enumerate(present_neighbours):
        if neighbour == head_id:
            v_conf[j]=head_confidence_data
        else:
            raw = metrics_map[neighbour].get('confidence_score')
            v_conf[j] = json.loads(raw) if raw is not None else 0.5

    # SIZE -- normalized by total across neighbours per class
    v_norm_size = []
    for j, neighbour in enumerate(present_neighbours):
        counts = np.array(effective_matrix[neighbour], dtype=float)

        # neighbour size < 3 and confident -> use head's size instead
        for i in range(class_number):
            if counts[i] < 3 and v_conf[j][i] != 0:
                new_val = effective_matrix[head_id][i]
                counts[i] = new_val
                effective_matrix[neighbour][i] = new_val

        total_per_class = np.sum([np.array(effective_matrix[n]) for n in present_neighbours], axis=0)
        normed = np.zeros_like(counts, dtype=float)
        for i in range(class_number):
            if total_per_class[i] != 0:
                normed[i] = counts[i] / total_per_class[i]

        v_norm_size.append(normed)


    # DISTANCE -- prob_matrix
    head_prob_matrix = np.array(json.loads(metrics_map[head_id]['prob_matrix'])).reshape(class_number, class_number)
    v_distance = np.ones((len(present_neighbours), class_number))

    for i, neighbour in enumerate(present_neighbours):
        neigh_prob_matrix = np.array(json.loads(metrics_map[neighbour]['prob_matrix'])).reshape(class_number, class_number)
        for j in range(class_number):
            current_has_class = has_class[head_id][j]
            neighbor_has_class = has_class[neighbour][j]
            if neighbour == head_id:
                dist = 1.0
            elif not current_has_class and neighbor_has_class:
                dist = 1.0
            elif not current_has_class or not neighbor_has_class:
                dist = 0.0
            else:
                dist = float(np.linalg.norm(
                    np.array(head_prob_matrix[j]) - np.array(neigh_prob_matrix[j])
                ))
            v_distance[i][j] = dist

    # SCORE + AGGREGATION
    raw_score = np.zeros((len(present_neighbours), class_number))

    for i, neighbour in enumerate(present_neighbours):
        for j in range(class_number):
            raw_score[i][j] = (v_distance[i][j] ** eff_alpha) * \
                              (v_conf[i][j] ** eff_beta)# * \
                              #(v_norm_size[i][j] ** eff_gamma)

    raw_score_sum = np.sum(raw_score, axis=0)
    v_score = np.divide(raw_score, raw_score_sum, out=np.zeros_like(raw_score), where=raw_score_sum > 0)

    global_score_client = np.sum(v_score, axis=1)
    sum_weights = np.sum(global_score_client)
    weights = global_score_client / sum_weights if sum_weights > 0 else np.ones(len(present_neighbours)) / len(present_neighbours)

    #log(INFO, "Aggregation Weights: %s", np.round(weights, 4))

    aggregated_params = [np.zeros_like(layer) for layer in parameters_to_ndarrays(ordered_results[0][1].parameters)]
    for w, (_, fit_res) in zip(weights, ordered_results):
        client_layers = parameters_to_ndarrays(fit_res.parameters)
        for idx, layer in enumerate(client_layers):
            aggregated_params[idx] += w * layer

    return aggregated_params


def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(num_examples for (num_examples, _) in results)
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    if num_total_evaluation_examples == 0:
        num_total_evaluation_examples = 1
    return sum(weighted_losses) / num_total_evaluation_examples

def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum(num_examples for (_, num_examples) in results)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

def aggregate_median(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute median."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    # Compute median weight of each layer
    median_w: NDArrays = [
        np.median(np.asarray(layer), axis=0) for layer in zip(*weights)
    ]
    return median_w


def aggregate_krum(
    results: List[Tuple[NDArrays, int]], num_malicious: int, to_keep: int
) -> NDArrays:
    """Choose one parameter vector according to the Krum function.

    If to_keep is not None, then MultiKrum is applied.
    """
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    # Compute distances between vectors
    distance_matrix = _compute_distances(weights)

    # For each client, take the n-f-2 closest parameters vectors
    num_closest = max(1, len(weights) - num_malicious - 2)
    closest_indices = []
    for distance in distance_matrix:
        closest_indices.append(
            np.argsort(distance)[1 : num_closest + 1].tolist()  
        )

    # Compute the score for each client, that is the sum of the distances
    # of the n-f-2 closest parameters vectors
    scores = [
        np.sum(distance_matrix[i, closest_indices[i]])
        for i in range(len(distance_matrix))
    ]

    if to_keep > 0:
        # Choose to_keep clients and return their average (MultiKrum)
        best_indices = np.argsort(scores)[::-1][len(scores) - to_keep :]  
        best_results = [results[i] for i in best_indices]
        return aggregate(best_results)

    # Return the model parameters that minimize the score (Krum)
    return weights[np.argmin(scores)]


# pylint: disable=too-many-locals
def aggregate_bulyan(
    results: List[Tuple[NDArrays, int]],
    num_malicious: int,
    aggregation_rule: Callable,  # type: ignore
    **aggregation_rule_kwargs: Any,
) -> NDArrays:
    """Perform Bulyan aggregation.

    Parameters
    ----------
    results: List[Tuple[NDArrays, int]]
        Weights and number of samples for each of the client.
    num_malicious: int
        The maximum number of malicious clients.
    aggregation_rule: Callable
        Byzantine resilient aggregation rule used as the first step of the Bulyan
    aggregation_rule_kwargs: Any
        The arguments to the aggregation rule.
    Returns
    -------
    aggregated_parameters: NDArrays
        Aggregated parameters according to the Bulyan strategy.
    """
    byzantine_resilient_single_ret_model_aggregation = [aggregate_krum]
    # also GeoMed (but not implemented yet)
    byzantine_resilient_many_return_models_aggregation = []  # type: ignore
    # Brute, Medoid (but not implemented yet)

    num_clients = len(results)
    if num_clients < 4 * num_malicious + 3:
        raise ValueError(
            "The Bulyan aggregation requires then number of clients to be greater or "
            "equal to the 4 * num_malicious + 3. This is the assumption of this method."
            "It is needed to ensure that the method reduces the attacker's leeway to "
            "the one proved in the paper."
        )
    selected_models_set: List[Tuple[NDArrays, int]] = []

    theta = len(results) - 2 * num_malicious
    beta = theta - 2 * num_malicious

    for _ in range(theta):
        best_model = aggregation_rule(
            results=results, num_malicious=num_malicious, **aggregation_rule_kwargs
        )
        list_of_weights = [weights for weights, num_samples in results]
        # This group gives exact result
        if aggregation_rule in byzantine_resilient_single_ret_model_aggregation:
            best_idx = _find_reference_weights(best_model, list_of_weights)
        # This group requires finding the closest model to the returned one
        # (weights distance wise)
        elif aggregation_rule in byzantine_resilient_many_return_models_aggregation:
            # when different aggregation strategies available
            # write a function to find the closest model
            raise NotImplementedError(
                "aggregate_bulyan currently does not support the aggregation rules that"
                " return many models as results. "
                "Such aggregation rules are currently not available in Flower."
            )
        else:
            raise ValueError(
                "The given aggregation rule is not added as Byzantine resilient. "
                "Please choose from Byzantine resilient rules."
            )

        selected_models_set.append(results[best_idx])

        # remove idx from tracker and weights_results
        results.pop(best_idx)

    # Compute median parameter vector across selected_models_set
    median_vect = aggregate_median(selected_models_set)

    # Take the averaged beta parameters of the closest distance to the median
    # (coordinate-wise)
    parameters_aggregated = _aggregate_n_closest_weights(
        median_vect, selected_models_set, beta_closest=beta
    )
    return parameters_aggregated


def aggregate_qffl(
    parameters: NDArrays, deltas: List[NDArrays], hs_fll: List[NDArrays]
) -> NDArrays:
    """Compute weighted average based on Q-FFL paper."""
    demominator: float = np.sum(np.asarray(hs_fll))
    scaled_deltas = []
    for client_delta in deltas:
        scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])
    updates = []
    for i in range(len(deltas[0])):
        tmp = scaled_deltas[0][i]
        for j in range(1, len(deltas)):
            tmp += scaled_deltas[j][i]
        updates.append(tmp)
    new_parameters = [(u - v) * 1.0 for u, v in zip(parameters, updates)]
    return new_parameters


def _compute_distances(weights: List[NDArrays]) -> NDArray:
    """Compute distances between vectors.

    Input: weights - list of weights vectors
    Output: distances - matrix distance_matrix of squared distances between the vectors
    """
    flat_w = np.array([np.concatenate(p, axis=None).ravel() for p in weights])
    distance_matrix = np.zeros((len(weights), len(weights)))
    for i, flat_w_i in enumerate(flat_w):
        for j, flat_w_j in enumerate(flat_w):
            delta = flat_w_i - flat_w_j
            norm = np.linalg.norm(delta)
            distance_matrix[i, j] = norm**2
    return distance_matrix


def _trim_mean(array: NDArray, proportiontocut: float) -> NDArray:
    """Compute trimmed mean along axis=0.
    It is based on the scipy implementation.
    https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.stats.trim_mean.html.
    """
    axis = 0
    nobs = array.shape[axis]
    lowercut = int(proportiontocut * nobs)
    uppercut = nobs - lowercut
    if lowercut > uppercut:
        raise ValueError("Proportion too big.")

    atmp = np.partition(array, (lowercut, uppercut - 1), axis)

    slice_list = [slice(None)] * atmp.ndim
    slice_list[axis] = slice(lowercut, uppercut)
    result: NDArray = np.mean(atmp[tuple(slice_list)], axis=axis)
    return result


def aggregate_trimmed_avg(
    results: List[Tuple[NDArrays, int]], proportiontocut: float
) -> NDArrays:
    """Compute trimmed average."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    trimmed_w: NDArrays = [
        _trim_mean(np.asarray(layer), proportiontocut=proportiontocut)
        for layer in zip(*weights)
    ]

    return trimmed_w


def _check_weights_equality(weights1: NDArrays, weights2: NDArrays) -> bool:
    """Check if weights are the same."""
    if len(weights1) != len(weights2):
        return False
    return all(
        np.array_equal(layer_weights1, layer_weights2)
        for layer_weights1, layer_weights2 in zip(weights1, weights2)
    )


def _find_reference_weights(
    reference_weights: NDArrays, list_of_weights: List[NDArrays]
) -> int:
    """Find the reference weights by looping through the `list_of_weights`.

    Raise Error if the reference weights is not found.

    Parameters
    ----------
    reference_weights: NDArrays
        Weights that will be searched for.
    list_of_weights: List[NDArrays]
        List of weights that will be searched through.

    Returns
    -------
    index: int
        The index of `reference_weights` in the `list_of_weights`.

    Raises
    ------
    ValueError
        If `reference_weights` is not found in `list_of_weights`.
    """
    for idx, weights in enumerate(list_of_weights):
        if _check_weights_equality(reference_weights, weights):
            return idx
    raise ValueError("The reference weights not found in list_of_weights.")


def _aggregate_n_closest_weights(
    reference_weights: NDArrays, results: List[Tuple[NDArrays, int]], beta_closest: int
) -> NDArrays:
    """Calculate element-wise mean of the `N` closest values.

    Note, each i-th coordinate of the result weight is the average of the beta_closest
    -ith coordinates to the reference weights


    Parameters
    ----------
    reference_weights: NDArrays
        The weights from which the distances will be computed
    results: List[Tuple[NDArrays, int]]
        The weights from models
    beta_closest: int
        The number of the closest distance weights that will be averaged

    Returns
    -------
    aggregated_weights: NDArrays
        Averaged (element-wise) beta weights that have the closest distance to
         reference weights
    """
    list_of_weights = [weights for weights, num_examples in results]
    aggregated_weights = []

    for layer_id, layer_weights in enumerate(reference_weights):
        other_weights_layer_list = []
        for other_w in list_of_weights:
            other_weights_layer = other_w[layer_id]
            other_weights_layer_list.append(other_weights_layer)
        other_weights_layer_np = np.array(other_weights_layer_list)
        diff_np = np.abs(layer_weights - other_weights_layer_np)
        indices = np.argpartition(diff_np, kth=beta_closest - 1, axis=0)
        beta_closest_weights = np.take_along_axis(
            other_weights_layer_np, indices=indices, axis=0
        )[:beta_closest]
        aggregated_weights.append(np.mean(beta_closest_weights, axis=0))
    return aggregated_weights

