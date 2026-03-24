#============================================================================== #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                           #
# All rights reserved.                                                          #
#                                                                               #
# This source code and the accompanying materials are made available under      #
# the terms of the Apache License 2.0 which accompanies this distribution.      #
# The QAOA-GPT implementation in CUDA-Q is based on this paper:                 #
# https://arxiv.org/pdf/2504.16350                                              #
# Usage or reference of this code or algorithms requires citation of the paper: #
# Ilya Tyagin, Marwa Farag, Kyle Sherbert, Karunya Shirali, Yuri Alexeev,       #
# Ilya Safro "QAOA-GPT: Efficient Generation of Adaptive and Regular Quantum    #
# Approximate Optimization Algorithm Circuits", IEEE International Conference   #
# on Quantum Computing and Engineering (QCE), 2025.                             #
# ============================================================================= #

import random
import numpy as np
import networkx as nx


def _is_independent_set(graph, node_set):
    """Check that no two nodes in node_set are adjacent."""
    for u in node_set:
        for v in node_set:
            if u < v and graph.has_edge(u, v):
                return False
    return True


def brute_force_mis(graph):
    """
    Computes the Maximum Independent Set using brute-force enumeration.

    Args:
        graph (nx.Graph): The input graph.

    Returns:
        tuple: (-mis_size, selected_set, binary_string)
            Energy is negative because adapt_qaoa_run minimizes.
    """
    nodes = list(graph.nodes)
    n = len(nodes)
    best_size = 0
    best_set = set()

    for i in range(2**n):
        binary = bin(i)[2:].zfill(n)
        selected = {nodes[j] for j, bit in enumerate(binary) if bit == '1'}
        if _is_independent_set(graph, selected) and len(selected) > best_size:
            best_size = len(selected)
            best_set = selected

    binary_string = ''.join('1' if node in best_set else '0' for node in nodes)
    return (-best_size), best_set, binary_string


def greedy_mis(graph):
    """
    Computes an approximate MIS using NetworkX's greedy heuristic.

    Args:
        graph (nx.Graph): The input graph.

    Returns:
        tuple: (-mis_size, selected_set, binary_string)
    """
    nodes = list(graph.nodes)
    mis_set = set(nx.approximation.maximum_independent_set(graph))
    binary_string = ''.join('1' if node in mis_set else '0' for node in nodes)
    return (-len(mis_set)), mis_set, binary_string


def simulated_annealing_mis(graph, initial_temp=1000, cooling_rate=0.95,
                            iterations=1000):
    """
    Computes an approximate MIS using simulated annealing.

    Args:
        graph (nx.Graph): The input graph.
        initial_temp (float): Starting temperature.
        cooling_rate (float): Multiplicative cooling factor.
        iterations (int): Number of SA steps.

    Returns:
        tuple: (-mis_size, selected_set, binary_string)
    """
    nodes = list(graph.nodes)
    n = len(nodes)

    # Start with an empty set and greedily add non-adjacent nodes
    current = [0] * n
    for idx in range(n):
        node = nodes[idx]
        neighbors_selected = any(
            current[nodes.index(nb)] for nb in graph.neighbors(node)
        )
        if not neighbors_selected:
            current[idx] = 1

    def mis_value(sol):
        """Return MIS size if valid, else size minus a large penalty."""
        selected = {nodes[j] for j, bit in enumerate(sol) if bit == 1}
        violations = sum(
            1 for u in selected for v in selected
            if u < v and graph.has_edge(u, v)
        )
        return len(selected) - violations * n  # heavy penalty per violation

    current_val = mis_value(current)
    best = current[:]
    best_val = current_val
    temp = initial_temp

    for _ in range(iterations):
        idx = random.randrange(n)
        new_sol = current[:]
        new_sol[idx] = 1 - new_sol[idx]
        new_val = mis_value(new_sol)

        delta = new_val - current_val
        if delta > 0 or (temp > 0 and random.random() < np.exp(delta / temp)):
            current = new_sol
            current_val = new_val
            if new_val > best_val:
                best = new_sol[:]
                best_val = new_val

        temp *= cooling_rate

    best_set = {nodes[j] for j, bit in enumerate(best) if bit == 1}
    # Repair: remove violating nodes (keep the one with fewer neighbors)
    for u, v in list(graph.edges):
        if u in best_set and v in best_set:
            if graph.degree(u) >= graph.degree(v):
                best_set.discard(u)
            else:
                best_set.discard(v)

    binary_string = ''.join('1' if node in best_set else '0' for node in nodes)
    return (-len(best_set)), best_set, binary_string
