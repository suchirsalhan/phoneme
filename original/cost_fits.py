#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 11:43:04 2025

@author: fermin
"""

from typing import Sequence, Optional
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

def compute_maxent_from_matrix(F, target_expectations, tol=1e-6):
    """
    Compute maximum entropy distribution from a precomputed feature matrix.
    
    Parameters:
        F: numpy array of shape (n, m) where n = number of support points, m = number of features
        target_expectations: array of target expectations of shape (m,)
        tol: optimization tolerance
        
    Returns:
        p: numpy array of probabilities of shape (n,)
        lambdas: optimized Lagrange multipliers of shape (m,)
    """
    F = np.asarray(F)
    n, m = F.shape
    target_expectations = np.asarray(target_expectations)
    
    def neg_dual(lambdas):
        log_probs = np.dot(F, lambdas)
        logZ = logsumexp(log_probs)
        return logZ - np.dot(lambdas, target_expectations)

    def grad_neg_dual(lambdas):
        log_probs = np.dot(F, lambdas)
        logZ = logsumexp(log_probs)
        probs = np.exp(log_probs - logZ)
        expected_features = np.dot(probs, F)
        return expected_features - target_expectations

    lambdas0 = np.zeros(m)
    res = minimize(neg_dual, lambdas0, jac=grad_neg_dual, tol=tol)

    if not res.success:
        raise RuntimeError("Optimization failed:", res.message)

    lambdas = res.x
    log_probs = np.dot(F, lambdas)
    logZ = logsumexp(log_probs)
    probs = np.exp(log_probs - logZ)

    return probs, lambdas
