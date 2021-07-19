"""
File: fdt_verification_z3.py
Author(s): Jack Good
Created: Tue Jul 14 14:28:23 EDT 2021
Description: Verification of fuzzy decision trees using the SMT solver Z3.
Acknowledgements:
Copyright (c) 2021 Carnegie Mellon University
This code is subject to the license terms contained in the code repo.
"""

import numpy as np
from z3 import *
from time import time
from multiprocessing import Pool
from fdt_verification import VerificationDomain


# encodes the FDT f into z3 logic for the z3 solver s
# assumes the splitting function sigma(x) = min(0, max(1, x+0.5))
# returns z3 variables x and y representing the input and output of f
def fdt_to_z3(f, s):
    x = [Real("x%d" % i) for i in range(f.p)]
    y = [Real("y%d" % i) for i in range(f.q)]
    sigma = lambda x: If(x < -0.5, 0, If(x > 0.5, 1, x + 0.5))
    
    def encode(n):
        if n.is_leaf: return [v for v in n.v]
        split = sigma(sum([n.a[i]*x[i] for i in range(f.p)])+n.b)
        return [(1-split)*li + split*ri for li, ri, in zip(encode(n.l), encode(n.r))]
    
    e = encode(f)
    s.add([y[i] == e[i] for i in range(f.q)])
    return x, y


# verify a property a^T f(x) <= b on for FDT f on domain D
# only uses linear constraints from D
# optional stopping condition timeout: stop when timeout is reached (in seconds)
# returns truth value (or None if unable to determine) and info
# info is a dictionary with entries:
#   time: the time in seconds for the process to stop
#   counterexample: a point violating a^T f(x) <= b (only if the property is violated)
def verify_fdt_z3(f, D, a, b, timeout=None):
        info = dict()
        start = time()
        s = Solver()
        x, y = fdt_to_z3(f, s)
        for ai, bi in zip(D.A, D.b):
            s.add(sum([ai[i]*x[i] for i in range(f.p)]) <= bi)
        s.add(sum([a[i]*y[i] for i in range(f.q)]) > b)
        if timeout: s.set("timeout", int(1000*timeout))
        c = s.check()
        info["time"] = time() - start
        if c == sat:
            # get value from model
            m = s.model()
            def v(x): return m[x].numerator().as_long()/m[x].denominator().as_long()
            info["counterexample"] = np.array([v(xi) for xi in x])
            return False, info
        if c == unsat:
            return True, info
        return None, info

    
# check local adversarial robustness of f at point(s) x with radius delta
# uses the L-infinity norm for distance
# optional stopping condition timeout: stop when timeout is reached (in seconds)
# returns:
#   robust: whether each case is proven robust
#   success: whether each case was successfully checked
#   counterexamples: a counterexample to robustness for each case (only when success but not robust)
#   times: the time spent to test each case in seconds
def check_local_robustness_z3(f, x, delta, timeout=None):
    onesample = False
    if x.ndim == 1: 
        onesample = True
        x = x.reshape(1,-1)
    if x.ndim != 2: raise ValueError("check_local_robustness_z3: x must be 1- or 2-dimensional")
    y = f(x)
    y = np.argmax(y, axis=1)
    robust = np.ones(x.shape[0], dtype=bool)
    success = np.ones_like(robust, dtype=bool)
    counterexamples = np.zeros_like(x)
    times = np.zeros_like(robust)
    for i, (xi, yi) in enumerate(zip(x, y)):
        start = time()
        D = VerificationDomain(xi.shape[0]).ball_linf(xi, delta)
        # must check every other label
        aa = np.zeros((f.q-1, f.q))
        aa[:yi,:yi] = np.eye(yi)
        aa[yi:,yi+1:] = np.eye(f.q-yi-1)
        aa[:,yi] = -1
        for a in aa:
            to = None if timeout is None else timeout+start-time()
            holds, info = verify_fdt_z3(f, D, a, b=0, timeout=to)
            if holds is None or not holds:
                robust[i] = False
                success[i] = holds is not None
                if success[i]: counterexamples[i] = info["counterexample"]
        times[i] = time() - start
    if onesample: return robust[0], success[0], counterexamples[0], times[0]
    return robust, success, counterexamples, times


# find adversarial perturbation of point(s) x on FDT f which is minimal within tolerance tol
# uses binary search over the radius of perturbation
# uses the L-infinity norm for distance
# n_workers specifies the number of processes to assign the tests to (default 1)
# optional stopping conditions:
#   timeout: stop when timeout is reached (in seconds)
#   upper_lim: stop if a perturbation within radius upper_lim cannot be found
# returns:
#   perturbation: the found perturbations (if not found, 0 vector is returned)
#   dist: the lower bounds of the distances to the perturbations, i.e. maximum provably robust radius
#   exact: whether each result is within tolerance
#   times: the time to compute each result in seconds
def minimum_adversarial_perturbation_z3(f, x, tol, timeout=None, upper_lim=None, n_workers=1):
    onesample = x.ndim == 1
    if onesample: x = x.reshape(1,-1)
    if x.shape[0] > 1 and n_workers > 1:
        with Pool(n_workers) as p:
            results = p.starmap(minimum_adversarial_perturbation_z3, \
                                [(f, xi, tol, timeout, upper_lim, 1) for xi in x])
        return tuple([np.array([r[i] for r in results]) for i in range(4)])
    perturbation = np.zeros_like(x)
    dist = np.zeros(x.shape[0])
    exact = np.zeros_like(dist, dtype=bool)
    times = np.zeros_like(dist)
    for i, xi in enumerate(x):
        start = time()
        lo = 0
        hi = tol
        while (upper_lim is None or hi < upper_lim) and (timeout is None or time() - start <= timeout):
            d = hi
            to = None if timeout is None else timeout+start-time()
            v, s, ce, _ = check_local_robustness_z3(f, xi, d, to)
            if s and not v: 
                exact[i] = True
                perturbation[i] = ce
                break
            if v: lo = hi
            hi *= 2
        if upper_lim is not None and hi >= upper_lim:
            dist[i] = upper_lim
            times[i] = time() - start
            continue
        while hi-lo > tol:
            if timeout and time() - start > timeout:
                exact[i] = False
                break
            d = (hi+lo)/2
            to = None if timeout is None else timeout+start-time()
            v, s, ce, _ = check_local_robustness_z3(f, xi, d, to)
            if not v:
                hi = d
                exact[i] = s
                if s: perturbation[i] = ce
            else: 
                lo = d
        dist[i] = lo # TODO: set this to hi and return the minimal perturbation
        times[i] = time() - start
    if onesample: return perturbation[0], dist[0], exact[0], times[0]
    return perturbation, dist, exact, times