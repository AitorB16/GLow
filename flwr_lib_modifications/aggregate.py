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
"""Aggregation functions for strategy implementations."""
# mypy: disallow_untyped_calls=False

from functools import reduce
from typing import Any, Callable, List, Tuple

import numpy as np

from flwr.common import FitRes, NDArray, NDArrays, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from scipy.spatial.distance import pdist, cdist, squareform, euclidean, cosine


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


def aggregate_inplace(results: List[Tuple[ClientProxy, FitRes]]) -> NDArrays:
    """Compute in-place weighted average."""
    # Count total examples
    num_examples_total = sum(fit_res.num_examples for (_, fit_res) in results)
    
    # DETECT IF IS GREATER THAN 0 (i.e., NODE HAS LOCAL INSTANCES) AVOID DIVISION BY 0 (ISLANDS)
    if num_examples_total > 0:
    
    # Compute scaling factors for each result
        scaling_factors = [
            fit_res.num_examples / num_examples_total for _, fit_res in results
        ]
    else:
        scaling_factors = [
            1. for _, fit_res in results
        ]

    # Let's do in-place aggregation
    # Get first result, then add up each other
    params = [
        scaling_factors[0] * x for x in parameters_to_ndarrays(results[0][1].parameters)
    ]
    for i, (_, fit_res) in enumerate(results[1:]):
        res = (
            scaling_factors[i + 1] * x
            for x in parameters_to_ndarrays(fit_res.parameters)
        )
        params = [reduce(np.add, layer_updates) for layer_updates in zip(params, res)]
    return params


def aggregate_score(results: List[Tuple[ClientProxy, FitRes]], neighbour_metrics: List[float], neighbours: List[int], head_id: int) -> NDArrays:
    """Compute score weighted average using test-sets / common challenge."""

    ordered_results = []
    for n in neighbours:
        for i, (cli, fit_res) in enumerate(results):
            if n == cli.cid:
                #print(neighbours)
                #print(cli.cid)
                ordered_results.append(results[i])
        
    scaling_norm = 0.
    for i, (cli, fit_res) in enumerate(ordered_results):
        if cli.cid == head_id:
            scaling_norm += fit_res.metrics['acc_val_distr']
        else:
            if neighbour_metrics[i] is not None:
                scaling_norm += neighbour_metrics[i]

    # AVOID DIVISION BY 0
    if scaling_norm == 0:
        scaling_norm = 1.0

    scaling_factors = []
    for i, (cli, fit_res) in enumerate(ordered_results):
        if cli.cid == head_id:
            scaling_factors.append(fit_res.metrics['acc_val_distr'] / scaling_norm)
        else:
            if neighbour_metrics[i] is not None:
                scaling_factors.append(neighbour_metrics[i] / scaling_norm)
            else:
                scaling_factors.append(0.)

    # Let's do in-place aggregation
    # Get first result, then add up each other

    params = [
        scaling_factors[0] * x for x in parameters_to_ndarrays(ordered_results[0][1].parameters)
    ]
    
    for i, (_, fit_res) in enumerate(ordered_results[1:]):
        res = (
            scaling_factors[i + 1] * x for x in parameters_to_ndarrays(fit_res.parameters)
        )
        params = [reduce(np.add, layer_updates) for layer_updates in zip(params, res)]
    
    return params

def aggregate_score_validation(results: List[Tuple[ClientProxy, FitRes]], neighbours: List[int], head_id: int) -> NDArrays:
    """Compute score weighted average using validation sets."""

    ordered_results = []
    for n in neighbours:
        for i, (cli, fit_res) in enumerate(results):
            if n == cli.cid:
                #print(neighbours)
                #print(cli.cid)
                ordered_results.append(results[i])
        
    scaling_norm = 0.
    for i, (cli, fit_res) in enumerate(ordered_results):
        scaling_norm += fit_res.metrics['acc_val_distr']

    # AVOID DIVISION BY 0
    if scaling_norm == 0:
        scaling_norm = 1.0

    scaling_factors = []
    for i, (cli, fit_res) in enumerate(ordered_results):
        scaling_factors.append(fit_res.metrics['acc_val_distr'] / scaling_norm)

    # Let's do in-place aggregation
    # Get first result, then add up each other

    params = [
        scaling_factors[0] * x for x in parameters_to_ndarrays(ordered_results[0][1].parameters)
    ]
    
    for i, (_, fit_res) in enumerate(ordered_results[1:]):
        res = (
            scaling_factors[i + 1] * x for x in parameters_to_ndarrays(fit_res.parameters)
        )
        params = [reduce(np.add, layer_updates) for layer_updates in zip(params, res)]
    
    return params

def aggregate_score_centroids_1(results: List[Tuple[ClientProxy, FitRes]], neighbour_metrics: List[float], neighbours: List[int], head_id: int, current_round: int, class_number: int, alpha: int = 0.5) -> NDArrays:
    """Compute centroids average 1."""

    alpha_prima = alpha
    #alpha = 0.5

    ordered_results = []
    for n in neighbours:
        for i, (cli, fit_res) in enumerate(results):
            if n == cli.cid:
                ordered_results.append(results[i])

    centroids = np.ones((len(neighbours), class_number))
    for i, (cli, fit_res) in enumerate(ordered_results):
        if cli.cid == head_id:
            centroids[i] = fit_res.metrics['centroid'][0] #TRAINING ONE
        else:
            centroids[i] = fit_res.metrics['centroid'] #NEIGHBORS

    dissimilarity_matrix = squareform(pdist(centroids, metric='cosine'))
    dissimilarity_vector = dissimilarity_matrix[neighbours.index(head_id)]
    dissimilarity_vector = np.clip(dissimilarity_vector, 1e-6, None)
    dissimilarity_vector = np.delete(dissimilarity_vector, neighbours.index(head_id)) #First row
    dissimilarity_vector_sum = sum(dissimilarity_vector)

    if dissimilarity_vector_sum > 0.:
        dissimilarity_vector = dissimilarity_vector/dissimilarity_vector_sum

    #I should now subsitute neigh_metrics
    scaling_norm = 0.
    for i, (cli, fit_res) in enumerate(ordered_results):
        if cli.cid == head_id:
            scaling_norm += fit_res.metrics['acc_val_distr']
        else:
            if neighbour_metrics[i] is not None:
                scaling_norm += neighbour_metrics[i]


    scaling_factors = []
    for i, (cli, fit_res) in enumerate(ordered_results):
        if cli.cid == head_id:
            if scaling_norm > 0.:
                scaling_factors.append(fit_res.metrics['acc_val_distr'] / scaling_norm)
            else:
                scaling_factors.append(0.)
        else:
            if neighbour_metrics[i] is not None and scaling_norm > 0.:
                scaling_factors.append(neighbour_metrics[i] / scaling_norm)
            else:
                scaling_factors.append(0.)

    # Let's do in-place aggregation
    # Get first result, then add up each other

    weights = []
    for i in range(len(ordered_results) - 1):
        w = (1 - alpha) * dissimilarity_vector[i] + alpha * scaling_factors[i + 1]
        weights.append(w)

    if dissimilarity_vector_sum > 0.:
        #print(dissimilarity_vector)
        all_weights = [alpha_prima * scaling_factors[0]] + list(weights)
        all_weights = np.array(all_weights)
        all_weights = all_weights / all_weights.sum()
    else:
        all_weights = scaling_factors

    #SELF PARAMS GO PLAIN
    params = [
        all_weights[0] * x for x in parameters_to_ndarrays(ordered_results[0][1].parameters)
    ]
    
    for i, (_, fit_res) in enumerate(ordered_results[1:]):
        res = (
            (all_weights[i + 1]) * x for x in parameters_to_ndarrays(fit_res.parameters)
        )
        params = [reduce(np.add, layer_updates) for layer_updates in zip(params, res)]
    
    return params

def aggregate_score_centroids_2(results: List[Tuple[ClientProxy, FitRes]], neighbours: List[int], head_id: int, current_round: int, class_client_matrix: List[List[int]], class_number: int = 10, alpha: float = 0.33, beta: float = 0.33, gamma: float = 0.33, val_ratio: float = 0.1) -> NDArrays:
    # !!!!!! ALPHA SOULD BE DYNAMIC? // MAYBE I NEED WARM-UP ROUNDS?? !!!!!!
    """Compute centroids average 2."""
    print(neighbours)

    ordered_results = []
    for n in neighbours:
        for i, (cli, fit_res) in enumerate(results):
            if n == cli.cid:
                ordered_results.append(results[i])
    
    #SIZE
    norm_terms_per_class = [0] * class_number
    for i in range(class_number):
        for neighbour in neighbours:
            if class_client_matrix[neighbour][i] > norm_terms_per_class[i]:
                norm_terms_per_class[i] = class_client_matrix[neighbour][i]
    #print(norm_terms_per_class)

    v_norm_size = []
    for neighbour in neighbours:
        v_norm_size.append(
            np.divide(
            np.array(class_client_matrix[neighbour]),
            norm_terms_per_class,
            out=np.zeros_like(norm_terms_per_class, dtype=float),
            where=np.array(norm_terms_per_class) > 0
            )
        )
    #print(v_norm_size)

    #CONFIDENCE
    for i, (cli, fit_res) in enumerate(ordered_results):
        if cli.cid == head_id:
            v_conf = fit_res.metrics['centroid']
            for j, (neighbour) in enumerate(neighbours):
                if neighbour != head_id:
                    for k in range(class_number):
                        if class_client_matrix[neighbour][k] == 0 and class_client_matrix[head_id][k] == 0:
                            v_conf[j][k] = 0. #Before was 1.
                        elif class_client_matrix[neighbour][k] == 0:
                            for l, (cli_tmp, fit_res_tmp) in enumerate(ordered_results):
                                if cli_tmp.cid == neighbour:
                                    tmp_centroid = fit_res_tmp.metrics['centroid']
                                    v_conf[j][k] = tmp_centroid[k]
    
    #print(v_conf)

    #COMPUTE PSEUDO-CENTROIDS BY SUMMING CENTROIDS AND DIVIDING THEM BY THE NUMBER OF INSTANACES PER CLASS

    #DISTANCE
    neigh_centroids = []
    class_client_validation = np.array(class_client_matrix[head_id] * val_ratio)
    pseudo_centroid = np.zeros(class_number) #NUM_CLASSES

    for i, (cli, fit_res) in enumerate(ordered_results):
        if cli.cid == head_id:
            neigh_centroids.append(fit_res.metrics['centroid'][0])
            pseudo_centroid += np.array(fit_res.metrics['centroid'][0])
        else:
            neigh_centroids.append(fit_res.metrics['centroid'])
            pseudo_centroid += np.array(fit_res.metrics['centroid'])
    
    pseudo_centroid = np.where(
        class_client_validation > 0,
        pseudo_centroid / class_client_validation,
        pseudo_centroid
    )

    v_distance = np.ones((len(neighbours), class_number))

    for i, (neighbour) in enumerate(neighbours):
        for j in range(class_number):
            if neighbour != head_id and class_client_matrix[neighbour][j] > 0 and class_client_matrix[head_id][j] > 0: # // CHECK |HEAD-PSEUDO_CNTR|
                v_distance[i][j] = abs(pseudo_centroid[j] - neigh_centroids[i][j])

    #print(v_distance)

    #SCORE -- ALPHA, BETA, GAMMA
    raw_score = np.zeros((len(neighbours), class_number))
    for i, (neighbour) in enumerate(neighbours):
        for j in range(class_number):
            raw_score[i][j] = (v_distance[i][j]**alpha) * (v_conf[i][j]**beta) * (v_norm_size[i][j]**gamma)
    
    raw_score_sum = np.zeros(class_number)
    for i in range(class_number):
        for j, (neighbour) in enumerate(neighbours):
            raw_score_sum[i] += raw_score[j][i]

    v_score = np.zeros((len(neighbours), class_number))
    for i, (neighbour) in enumerate(neighbours):
        v_score[i] = np.divide(
            raw_score[i],
            raw_score_sum,
            out=np.zeros_like(raw_score_sum, dtype=float),
            where=np.array(raw_score_sum) > 0
            )
    
    #print(v_score)

    #GLOBAL SCORE PER NEIGHBOR
    global_score_client = np.zeros(len(neighbours))
    for i, (neighbour) in enumerate(neighbours):
        global_score_client[i] = sum(v_score[i])

    weights = global_score_client.copy()
    sum_weights = sum(global_score_client)
    if sum_weights > 0:
        weights /= sum_weights

    print(weights)

    # Can I make a single loop?
    params = [
        weights[0] * x for x in parameters_to_ndarrays(ordered_results[0][1].parameters)
    ]
    
    for i, (_, fit_res) in enumerate(ordered_results[1:]):
        res = (
            (weights[i + 1]) * x for x in parameters_to_ndarrays(fit_res.parameters)
        )
        params = [reduce(np.add, layer_updates) for layer_updates in zip(params, res)]
    
    return params

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
            np.argsort(distance)[1 : num_closest + 1].tolist()  # noqa: E203
        )

    # Compute the score for each client, that is the sum of the distances
    # of the n-f-2 closest parameters vectors
    scores = [
        np.sum(distance_matrix[i, closest_indices[i]])
        for i in range(len(distance_matrix))
    ]

    if to_keep > 0:
        # Choose to_keep clients and return their average (MultiKrum)
        best_indices = np.argsort(scores)[::-1][len(scores) - to_keep :]  # noqa: E203
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


def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(num_examples for (num_examples, _) in results)
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    if num_total_evaluation_examples == 0:
        num_total_evaluation_examples = 1
    return sum(weighted_losses) / num_total_evaluation_examples


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
        # Create indices of the smallest differences
        # We do not need the exact order but just the beta closest weights
        # therefore np.argpartition is used instead of np.argsort
        indices = np.argpartition(diff_np, kth=beta_closest - 1, axis=0)
        # Take the weights (coordinate-wise) corresponding to the beta of the
        # closest distances
        beta_closest_weights = np.take_along_axis(
            other_weights_layer_np, indices=indices, axis=0
        )[:beta_closest]
        aggregated_weights.append(np.mean(beta_closest_weights, axis=0))
    return aggregated_weights
