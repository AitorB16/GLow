# Copyright 2020 Flower Labs GmbH. Modified for GLow.
# SPDX-License-Identifier: Apache-2.0
"""Vendored from flwr/server/strategy/aggregate.py, plus GLow's aggregators
(dispatched by GLow_vanilla.aggregate_fit):

- aggregate_inplace ('inplace'): sample-weighted average
- aggregate_score / aggregate_score_validation: weighted by test / self-reported accuracy
- aggregate_score_centroids_2 ('approach_2'), aggregate_score_centroids_2b ('approach_2b')
- aggregate_trimmed_mean ('trimmed_mean')
- aggregate_krum ('krum'): Flower's Krum/MultiKrum

`results` is {topology_index: FitRes}. The remaining Flower functions are unused.
"""

from functools import reduce
from typing import Dict, List, Tuple

import numpy as np
import json

from flwr.common import FitRes, NDArray, NDArrays, parameters_to_ndarrays


def self_learning(results: Dict[int, FitRes], neighbours: List[int], head_id: int) -> NDArrays:
    """The head's own model, unaggregated."""
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
    """Neighbours weighted by their test accuracy (`neighbour_metrics`), the
    head by its reported `acc_val_distr`."""

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
        # no accuracy signal: equal weights
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
    """Every participant weighted by its self-reported `acc_val_distr`."""

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
    """Per class score distance**alpha * confidence**beta (size**gamma off),
    normalized per class, summed into one weight per neighbour.
    distance: L2 between head/neighbour mean softmax rows on the head's val set.
    confidence: recall on own val set if it has the class, else on the head's."""

    eff_alpha, eff_beta, eff_gamma = alpha, beta, gamma

    ordered_results = [(n, results[n]) for n in neighbours if n in results]
    if head_id not in results:
        raise ValueError(f"Head {head_id}'s fit() result is missing.")
    present_neighbours = [n for n, _ in ordered_results]

    metrics_map = {idx: fit_res.metrics for idx, fit_res in ordered_results}

    effective_matrix = [list(row) for row in class_client_matrix]

    # train-set class coverage, before the size substitution below
    has_class = {n: [class_client_matrix[n][c] > 0 for c in range(class_number)] for n in present_neighbours}

    # CONFIDENCE
    v_conf = np.zeros((len(present_neighbours), class_number))
    for j, neighbour in enumerate(present_neighbours):
        m = metrics_map[neighbour]
        v_conf[j] = np.where(json.loads(m['own_val_has']), json.loads(m['recall_own']), json.loads(m['recall_head']))

    # SIZE: share of the class across neighbours
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


    # DISTANCE
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

    aggregated_params = [np.zeros_like(layer) for layer in parameters_to_ndarrays(ordered_results[0][1].parameters)]
    for w, (_, fit_res) in zip(weights, ordered_results):
        client_layers = parameters_to_ndarrays(fit_res.parameters)
        for idx, layer in enumerate(client_layers):
            aggregated_params[idx] += w * layer

    return aggregated_params


def aggregate_score_centroids_2b(
    results: Dict[int, FitRes],
    neighbours: List[int],
    head_id: int,
    class_client_matrix: List[List[int]],
    class_number: int,
    alpha: float = 0.4,
    beta: float = 0.6,
) -> NDArrays:
    """approach_2 with the val sets swapped:
    distance: n's `dist_N` (on n's own val set), gated as in approach_2.
    confidence: n's recall on the head's val set, or on its own for classes
    the head's val set lacks."""

    if head_id not in results:
        raise ValueError(f"Head {head_id}'s fit() result is missing.")
    ordered = [(n, results[n]) for n in neighbours if n in results]
    head_checks = np.array(json.loads(results[head_id].metrics['own_val_has']))
    head_has = np.array(class_client_matrix[head_id]) > 0

    v_conf = np.zeros((len(ordered), class_number))
    v_distance = np.ones((len(ordered), class_number))
    for i, (n, fit_res) in enumerate(ordered):
        m = fit_res.metrics
        recall_head = np.array(json.loads(m['recall_head']))
        if n == head_id:
            v_conf[i] = recall_head
            continue
        n_has = np.array(class_client_matrix[n]) > 0
        v_conf[i] = np.where(head_checks, recall_head, json.loads(m['recall_own']))
        v_distance[i] = np.where(head_has & n_has, json.loads(m['dist_N']), ~head_has & n_has)

    raw_score = (v_distance ** alpha) * (v_conf ** beta)
    raw_score_sum = raw_score.sum(axis=0)
    v_score = np.divide(raw_score, raw_score_sum, out=np.zeros_like(raw_score), where=raw_score_sum > 0)
    global_score = v_score.sum(axis=1)
    weights = global_score / global_score.sum() if global_score.sum() > 0 else np.ones(len(ordered)) / len(ordered)

    params = [np.zeros_like(layer) for layer in parameters_to_ndarrays(ordered[0][1].parameters)]
    for w, (_, fit_res) in zip(weights, ordered):
        for k, layer in enumerate(parameters_to_ndarrays(fit_res.parameters)):
            params[k] += w * layer
    return params


def aggregate_trimmed_mean(results: Dict[int, FitRes], trim: int = 1) -> NDArrays:
    """Per coordinate, drop the `trim` highest and lowest values and average
    the rest (trim shrinks with few models: 3 -> median, 1-2 -> mean)."""
    models = [parameters_to_ndarrays(results[idx].parameters) for idx in sorted(results)]
    k = min(trim, (len(models) - 1) // 2)
    params = []
    for layers in zip(*models):
        stacked = np.sort(np.stack(layers), axis=0)
        params.append(stacked[k:len(models) - k].mean(axis=0).astype(layers[0].dtype))
    return params


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
    results: Dict[int, FitRes], num_malicious: int, to_keep: int
) -> NDArrays:
    """Choose one parameter vector according to the Krum function.

    If to_keep > 0, then MultiKrum is applied.
    """
    results = [
        (parameters_to_ndarrays(results[idx].parameters), results[idx].num_examples)
        for idx in sorted(results)
    ]

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
        if sum(n for _, n in best_results) == 0:  # no chosen model holds data: equal weights
            best_results = [(w, 1) for w, _ in best_results]
        return aggregate(best_results)

    # Return the model parameters that minimize the score (Krum)
    return weights[np.argmin(scores)]


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
