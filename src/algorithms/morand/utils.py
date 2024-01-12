# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import numpy as np
from typing import List

def from_unit_cube(point, lb, ub):
    assert np.all(lb < ub) 
    assert lb.ndim == 1 
    assert ub.ndim == 1 
    assert point.ndim  == 2
    new_point = point * (ub - lb) + lb
    return new_point

def latin_hypercube(n, dims):
    points = np.zeros((n, dims))
    centers = (1.0 + 2.0 * np.arange(0.0, n)) 
    centers = centers / float(2 * n)
    for i in range(0, dims):
        points[:, i] = centers[np.random.permutation(n)]

    perturbation = np.random.uniform(-1.0, 1.0, (n, dims)) 
    perturbation = perturbation / float(2 * n)
    points += perturbation
    return points

# Multi-objective Utils for Pareto Calculation
def dominates_point(obj1, obj2):
    """
    Check if obj1 dominates obj2.

    Args:
        obj1 (list): A list of objective values for the first point.
        obj2 (list): A list of objective values for the second point.

    Returns:
        True if obj1 dominates obj2, False otherwise.
    """
    less_or_equal = all(a <= b for a, b in zip(obj1, obj2))
    less = any(a < b for a, b in zip(obj1, obj2))
    return less_or_equal and less

def fast_non_dominated_sort(objs):

    size = len(objs)
    pareto_frontier = []
    dominated = [0] * size
    dominates = [[] for _ in range(size)]

    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            if dominates_point(objs[i], objs[j]):
                dominates[i].append(j)
            elif dominates_point(objs[j], objs[i]):
                dominated[i] += 1
        if dominated[i] == 0:
            pareto_frontier.append(i)
    # print(objs)
    return pareto_frontier