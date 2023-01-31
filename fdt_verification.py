"""
File: fdt_verification.py
Author(s): Jack Good
Created: Tue Jul 14 14:28:23 EDT 2021
Description: Verification of fuzzy decision trees using our abstraction-refinement algorithm.
Acknowledgements:
Copyright (c) 2021 Carnegie Mellon University
This code is subject to the license terms contained in the code repo.
"""

import numpy as np
import cvxpy as cp
import heapq
from cvxpy.error import SolverError
from time import time
import multiprocessing


# set the solver
solver = None # let cvxpy decide


# a wrapper around cvxpy constraints with extra utility for this verification application
# using the fact that the number of possible linear constraint coefficients is limited,
# some redundant linear constraints are efficiently removed
class VerificationDomain():
    
    # initialize with dimension p
    def __init__(self, p):
        self.p = p
        self.x = cp.Variable(p)
        self.a = cp.Parameter(p)
        self.A = np.zeros((0,p))
        self.b = np.zeros(0)
        self.lin_map = dict() # map linear constraint coefficients to their index in A and b
        self.constraints = [None] # placeholder for the linear constraint
        self.interior_point = None # keep track of an interior point
        self.prob = None # compiled cvxpy problem
        self.needs_recompile = True # flag to recompile constraints on next solve

    # return deep copy of self
    def copy(self):
        other = VerificationDomain(self.p)
        other.x = self.x
        other.A = self.A.copy()
        other.b = self.b.copy()
        other.lin_map = self.lin_map.copy()
        other.constraints = self.constraints.copy()
        other.interior_point = self.interior_point
        return other
    
    # add a constraint to the domain
    # return self
    def constrain(self, constraint):
        self.constraints.append(constraint)
        self.needs_recompile = True
    
    # add linear constraints a^T x <= b 
    # automatically and efficiently reduces multiple constraints with the same a
    # return self
    def lin(self, a, b):
        if a.ndim == 1:
            a = a.reshape((1,-1))
            b = np.array([b])
        for ai, bi in zip(a, b):
            tup = tuple(ai)
            if tup in self.lin_map:
                j = self.lin_map[tup]
                self.b[j] = min(self.b[j], bi)
            else:
                self.lin_map[tup] = self.A.shape[0]
                self.A = np.vstack([self.A, ai])
                self.b = np.concatenate([self.b, np.array([bi])])
        self.constraints[0] = self.A@self.x <= self.b
        self.needs_recompile = True
        return self
    
    # add a box (hyperrectangle) constraint
    # argument box is a numpy array of size (p,2) with lower and upper bounds for each dimension
    # return self
    def box(self, box):
        A = np.concatenate((-np.eye(self.p), np.eye(self.p)))
        b = box.T.flatten().copy()
        b[:self.p] *= -1
        return self.lin(A, b)
    
    # add L-infinity ball constraint with radius delta about x
    # delta may be scalar or vector of size p
    # return self
    def ball_linf(self, x, delta):
        return self.box(np.stack((x-delta,x+delta), axis=1))

    # add L2 ball constraint with radius delta about x
    # delta may be scalar or vector of size p
    # return self
    def ball_l2(self, x, delta):
        self.constrain(sum(((self.x-x)/delta)**2) <= 1)
        return self

    # add L1 ball constraint with radius delta about x
    # delta may be scalar or vector of size p
    # the number of constraints is exponential in p, so be careful
    # return self
    def ball_l1(self, x, delta):
        a = np.ones(self.p)/delta
        A = np.zeros((2**self.p, self.p))
        i = 0
        def generate_constraints(j):
            nonlocal i
            if j == self.p:
                A[i] = a
                i += 1
            else:
                for _ in range(2):
                    generate_constraints(j+1)
                    a[j] = -a[j]
        generate_constraints(0)
        return self.lin(A, A@x+1)

    # partition domain on a^T x = b 
    # return the two resulting domains
    def split(self, a, b):
        return self.copy().lin(a, b), self.copy().lin(-a, -b)
    
    # test whether each point in x lies in the domain
    # this is slow because of evaluating one point at a time through cvxopt
    # return boolean array with same length as x
    def contains(self, x):
        onesample = False
        if x.ndim == 1: 
            onesample = True
            x = x.reshape(1,-1)
        sat = np.ones(x.shape[0], dtype=bool)
        for i, xi in enumerate(x):
            self.x.value = xi
            for c in self.constraints:
                if c is not None and not c.value():
                    sat[i] = False
                    break
        return sat
    
    # returns a point inside the domain, or None if the solver fails
    # if a is provided, it will return the point that maximizes a^T x
    # the result may be very slightly outside the domain due to numerical precision of the solver
    def get_interior_point(self, a=None):
        if a is not None: self.maximize(a)
        elif self.needs_recompile or self.interior_point is None: self.maximize(np.zeros(self.p))
        return self.interior_point
    
    # returns max a^T x over this domain, or None if the solver fails
    # the solver fails if, for example, the domain is empty or unbounded
    # optionally a timeout in seconds can be provided
    def maximize(self, a, timeout=None):
        if self.needs_recompile:
            self.prob = cp.Problem(cp.Maximize(self.a@self.x), \
                    self.constraints[1:] if self.constraints[0] is None else self.constraints)
            self.needs_recompile = False
            self.interior_point = None
        self.a.value = a
        self.prob.solve(solver=solver)
        if self.prob.status.startswith("optimal"):
            self.interior_point = self.x.value.copy()
            return self.prob.value
        raise SolverError("solver failed with status: " + self.prob.status)

# a linear splitting function
def linear_split(x): return np.maximum(0, np.minimum(1, x+0.5))

# the inverse of the linear split on [0,1]
def inv_linear_split(x): return x-0.5

# a collection of zero leaves, used in FDTNode to reduce redundancy
zeros = dict()

# a data structure for Fuzzy Decision Trees (FDT)
# also represents Fuzzy Decision Directed Acyclic Graphs (FDD)
# to verify your model, convert it to this structure
class FDTNode():
    
    # initialize as internal node with 
    # a the linear weight
    # b the linear intercept
    # s the splitting function, AKA sigma
    # t the inverse of s, or a valid substitute (see the paper for details)
    # l the left child
    # r the right child
    # eps a small value used for dynamic pruning (see the paper for details)
    # return the FDTNode
    def internal(a, b, s, t, l, r, eps=0):
        n = FDTNode()
        n.is_leaf = False
        n.a = a
        n.b = b
        n.s = s
        n.t = t
        n.l = l
        n.r = r
        n.eps = eps
        n.p = a.shape[0] # input dimension
        n.q = l.q # output dimension
        if l.q != r.q:
            raise ValueError("attempt to construct a node with incompatible children")
        n.memo = None # memoized information (to process DAGs efficiently)
        return n
    
    # initialize as leaf with v the value at this leaf
    # return the FDTNode
    def leaf(v):
        n = FDTNode()
        n.is_leaf = True
        n.v = v
        n.a = None # bounding coefficients
        n.av = 0 # value to bound
        n.q = v.shape[0]
        n.memo = None
        return n
    
    # initialize as a leaf with uniform output with value v and size q
    # return self
    def const(q, v):
        a = np.zeros(q)
        a[0] = 1
        return FDTNode.leaf(v*np.ones(q)).set_bounding_coefficients(a)
    
    # set the value of a for the bounding of a^T f(x)
    # must be called on a new initialized tree before bound
    # do not use this on FDTNodes constructed with FDTNode.add, FDTNode.mult, etc.
    # return self
    def set_bounding_coefficients(self, a):
        self.memo = None # this invalidates memoization
        if self.is_leaf:
            self.a = a
            self.av = np.dot(self.a, self.v)
        else:
            self.l.set_bounding_coefficients(a)
            self.r.set_bounding_coefficients(a)
        return self
    
    # evaluate this FDTNode at x and return the result
    # for classifiers, returns probability vector, not label
    # x may be of size (p) or (n,p) for some number of inputs n
    # top is a flag used internally to manage memoization
    def __call__(self, x, top=True):
        if top: self.forget()
        onesample = False
        if x.ndim == 1: 
            onesample = True
            x = x.reshape(1,-1)
        if self.memo is not None:
            y = self.memo
        elif self.is_leaf:
            y = np.repeat(self.v[None,:], x.shape[0], axis=0)
            self.memo = y
        else:
            s = self.s(x@self.a + self.b)
            if self.eps is not None:
                s[s < self.eps] = 0
                s[s > 1-self.eps] = 1
            y = (1-s[:,None])*self.l(x, False) + s[:,None]*self.r(x, False)
            self.memo = y
        if onesample: return y[0]
        return y
    
    # return depth and number of nodes
    # top is a flag used internally to manage memoization
    def size(self, top=True):
        if top: self.forget()
        if self.memo is not None: return self.memo[0], 0
        if self.is_leaf:
            self.memo = (0, 1)
        else:
            ld, ln = self.l.size(False)
            rd, rn = self.r.size(False)
            self.memo = (max(ld, rd)+1, ln+rn+1)
        return self.memo
   
    # compute bounds of a^T f(x) on domain D, where f is this FDTNode
    # top is a flag used internally to manage memoization
    # returns:
    #   l the lower bound
    #   u the upper bound
    #   s the best split (v, a, b), where v is the value of the split and a, b are the coefficients and intercept of the split
    #   g the dynamically pruned copy of f
    # see the paper for details on split value v
    def bound(self, D, top=True):
        if top: self.forget()
            
        # check for memoization
        if self.memo is not None:
            return self.memo
        
        # leaves just return the value
        if self.is_leaf:
            self.memo = (self.av, self.av, None, self)
            return self.memo
        
        # compute bounds (see the paper for details)
        smin = self.s(-D.maximize(-self.a) + self.b)
        smax = self.s( D.maximize( self.a) + self.b)
        sbar = 0.5*(smin + smax)
        if self.eps and smax < self.eps: # check if we can remove right subtree
            self.memo = self.l.bound(D, False)
            return self.memo
        if self.eps and 1-smin < self.eps:  # check if we can remove left subtree
            self.memo = self.r.bound(D, False)
            return self.memo
        lL, uL, sL, fL = self.l.bound(D, False)
        lR, uR, sR, fR = self.r.bound(D, False)
        l0 = (1-smin)*lL + smin*lR
        l1 = (1-smax)*lL + smax*lR
        u0 = (1-smin)*uL + smin*uR
        u1 = (1-smax)*uL + smax*uR
        l = min(l0, l1)
        u = max(u0, u1)
        
        # compute split for this node
        v = 0.5*(abs(l0 - l1) + abs(u0 - u1))
        anorm = np.linalg.norm(self.a)
        if anorm == 0: anorm = 1
        a, b = self.a/anorm, (self.t(sbar) - self.b)/anorm
        s = (v, a, b)
        
        # find best split so far
        if sL is not None:
            vL = (1-sbar)*sL[0]
            if vL > s[0]: s = (vL, sL[1], sL[2])
        if sR is not None:
            vR = sbar*sR[0]
            if vR > s[0]: s = (vR, sR[1], sR[2])
        
        # create pruned version of self by using pruned subtrees as children
        g = FDTNode.internal(self.a, self.b, self.s, self.t, fL, fR, self.eps)
        
        # memoize results
        self.memo = (l, u, s, g)
        return self.memo
    
    # return deep copy of self
    def copy(self):
        if self.is_leaf: 
            l = FDTNode.leaf(self.v.copy())
            if self.a is not None: l.set_bounding_coefficients(self.a)
            return l
        l, r = self.l.copy(), self.r.copy()
        return FDTNode.internal(self.a.copy(), self.b, self.s, self.t, l, r, self.eps)
    
    # clear memoization
    def forget(self):
        self.memo = None
        if not self.is_leaf:
            self.l.forget()
            self.r.forget()    
    
    # return constructed FDTNode f+g, where g is another FDTNode or a scalar
    def add(self, g):
        if not isinstance(g, FDTNode): g = FDTNode.const(self.q, g)
        return FDTNode.internal(np.zeros(self.p), 0.5, lambda x: x, lambda x: x, self.mult(2), g.mult(2))
    
    # return constructed FDTNode fg, where g is another FDTNode or a scalar
    # z is an appropriate zero leaf, used internally to reduce redundancy
    def mult(self, g):
        if self.is_leaf:
            if isinstance(g, FDTNode):
                if self.a is None: 
                    raise ValueError("bounding coefficients must be set to multiply by another FDTNode")
                return g.mult_sigma(np.zeros(self.q), lambda x: 0*x + self.av, lambda x: 0*x)
            fg = FDTNode.leaf(g*self.v)
            if self.a is not None: fg.set_bounding_coefficients(self.a)
            return fg
        return FDTNode.internal(self.a.copy(), self.b, self.s, self.t, self.l.mult(g), self.r.mult(g), self.eps)
    
    # return constructed FDTNode s(a^T x)f(x), where s is a nondecreasing function
    # t is the inverse of s, or a valid substitute (see the paper for details)
    def mult_sigma(self, a, s, t):
        if self.q not in zeros: zeros[self.q] = FDTNode.const(self.q, 0)
        return FDTNode.internal(a, 0, s, t, zeros[self.q], self, None)

    
# transform an FDTNode f for global adversarial robustness testing
# returns the transformed FDTNode
# see the paper for details
def global_adversarial_robustness_transform(f):
    f = f.copy()
    q = [f]
    while q:
        n = q.pop()
        if not n.is_leaf:
            q.append(n.l)
            q.append(n.r)
            n.a = np.concatenate((n.a, n.a))
    g = f.copy()
    q = [f]
    while q:
        n = q.pop()
        if n.is_leaf:
            n.v *= 2
        else:
            q.append(n.l)
            q.append(n.r)
            n.a[f.p:] *= 0
    q = [g]
    while q:
        n = q.pop()
        if n.is_leaf:
            n.v *= -2
        else:
            q.append(n.l)
            q.append(n.r)
            n.a[:f.p] *= 0
    return FDTNode.internal(np.zeros(2*f.p), 0.5, lambda x: x, lambda x: x, f, g)
 
    
# a key-value pair for max heaps used with the heapq package
class KV():
    # initialize with key k and value v
    def __init__(self, k, v):
        self.k = k
        self.v = v
    # compare to another key-value pair
    def __lt__(self, other):
        return other.k < self.k # inverted since heapq is a min heap   

# either maximize or verify a property of FDTNode f on domain D
# coefficients must be set for all leaves using FDTNode.set_bounding_coefficients
# at least one stopping condition must be specified:
#   b: stop when the truth of a^T f(x) <= b is determined
#   tol: stop when the maximum value is known within value tol
#   timeout: stop when timeout is reached (in seconds)
#   max_it: stop after a number of iterations
# if b is provided, returns truth value (or None if unable to determine) and info;
# otherwise, lower bound of maximum, upper bound of maximum, and info
# info is a dictionary with entries:
#   status: string indicating why the process stopped
#   time: the time in seconds for the process to stop
#   iterations: the number of iterations before stopping
#   lbound: list of the lower bound at each iteration
#   ubound: list of the upper bound at each iteration
#   x: the maximizing input, or a point violating a^T f(x) <= b (only if the property is violated)
def verify_fdt(f, D, b=None, tol=None, timeout=None, max_it=None):
        if b is None and tol is None and timeout is None and max_it is None:
            raise ValueError("requires at least one stopping condition (b, tol, timeout, max_it)")
        start = time()
        info = dict()
        info["lbound"] = []
        info["ubound"] = []
        try:
            L, U, s, f = f.bound(D)
        except SolverError:
            info["status"] = "solver failed"
            info["iterations"] = 1
            info["time"] = time() - start
            if b is None: return None, None, info
            else: return None, info
        i, j = 1, 1
        h = []
        if b is None or L > b: info["x"] = D.get_interior_point()
        while True:
            # record the bounds at this iteration
            info["lbound"].append(L)
            info["ubound"].append(U)
            
            # check stopping conditions
            if b is not None and (U <= b or L > b):
                info["status"] = "success"
                break
            if tol is not None and U - L < tol:
                info["status"] = "success"
                break
            if timeout and time() - start > timeout:
                info["status"] = "timeout"
                break
            if max_it and i >= max_it:
                info["status"] = "iteration limit reached"
                break
            
            try: # catch solver failures
                # split the domain and bound the results
                if s is not None: # we might reduce down to just a leaf and no split can be done
                    for E in D.split(*s[1:]):
                        l, u, s, g = f.bound(E)
                        L = max(L, l)
                        heapq.heappush(h, KV(u, (l, E, s, g, j+1)))
                        if b is not None and l > b: break
            except SolverError:
                info["status"] = "solver failed"
                break
                
            kv = heapq.heappop(h)
            l, D, s, f, j = kv.v
            U = kv.k
            if b is None or L > b: info["x"] = D.get_interior_point()
            i += 1
        info["iterations"] = i
        info["time"] = time() - start
        if b is None: # maximization
            return L, U, info
        else: # verification
            if U <= b: return True, info
            if L > b: return False, info
            return None, info

        
# check local adversarial robustness of f at point(s) x with radius delta using given norm for distance
# norm may be "inf", "l2", or "l1" (default "inf")
# optional stopping conditions:
#   timeout: stop when timeout is reached (in seconds)
#   max_it: stop after a number of iterations
# returns:
#   robust: whether each case is proven robust
#   success: whether each case was successfully checked
#   counterexample: a counterexample to robustness for each case (only when success but not robust)
#   times: the time spent to test each case in seconds
def check_local_robustness(f, x, delta, timeout=None, max_it=None, norm="inf"):
    onesample = False
    if x.ndim == 1: 
        onesample = True
        x = x.reshape(1,-1)
    y = f(x)
    y = np.argmax(y, axis=1)
    robust = np.ones(x.shape[0], dtype=bool)
    success = np.ones_like(robust)
    counterexample = np.zeros_like(x)
    times = np.zeros(x.shape[0])
    for i, (xi, yi) in enumerate(zip(x, y)):
        start = time()
        D = VerificationDomain(xi.shape[0])
        if norm == "inf" or norm == "infinity" or norm == "linf" or norm == "Linf":
            D.ball_linf(xi, delta)
        elif norm == "2" or norm == "l2" or norm == "L2" or norm == 2:
            D.ball_l2(xi, delta)
        elif norm == "1" or norm == "l1" or norm == "L1" or norm == 1:
            D.ball_l1(xi, delta)
        else: raise ValueError("unknown norm %s" + (norm))
        # must check every other label
        aa = np.zeros((f.q-1, f.q))
        aa[:yi,:yi] = np.eye(yi)
        aa[yi:,yi+1:] = np.eye(f.q-yi-1)
        aa[:,yi] = -1
        for a in aa:
            to = None if timeout is None else timeout+start-time()
            f.set_bounding_coefficients(a)
            holds, info = verify_fdt(f, D, b=0, max_it=max_it, timeout=to)
            if holds is None or not holds:
                robust[i] = False
                success[i] = holds is not None
                if success[i]: counterexample[i] = info["x"]
        times[i] = time() - start
    if onesample: return robust[0], success[0], counterexample[0], times[0]
    return robust, success, counterexample, times
            

# find adversarial perturbation of point(s) x on FDTNode f which is minimal on given norm within tolerance tol
# uses binary search over the radius of perturbation
# norm may be "inf", "l2", or "l1" (default "inf")
# n_workers specifies the number of processes to assign the tests to (default 1)
# optional stopping conditions:
#   timeout: stop when timeout is reached (in seconds)
#   upper_lim: stop if a perturbation within radius upper_lim cannot be found
# returns:
#   perturbation: the found perturbations (if not found, 0 vector is returned)
#   dist: the lower bounds of the distances to the perturbations, i.e. maximum provably robust radius
#   exact: whether each result is within tolerance
#   times: the time to compute each result in seconds
def minimum_adversarial_perturbation(f, x, tol, norm="inf", timeout=None, upper_lim=None, n_workers=1):
    onesample = x.ndim == 1
    if onesample: x = x.reshape(1,-1)
    if x.shape[0] > 1 and n_workers > 1:
        with multiprocessing.Pool(n_workers) as p:
            results = p.starmap(minimum_adversarial_perturbation, \
                                [(f, xi, tol, norm, timeout, upper_lim, 1) for xi in x])
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
            to = None if timeout is None else timeout+start-time()
            v, s, ce, _ = check_local_robustness(f, xi, hi, norm=norm, timeout=to)
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
            v, s, ce, _ = check_local_robustness(f, xi, d, norm=norm, timeout=to)
            if not v:
                hi = d
                exact[i] = s
                if s: perturbation[i] = ce
            else: 
                lo = d
        dist[i] = lo
        times[i] = time() - start
    if onesample: return perturbation[0], dist[0], exact[0], times[0]
    return perturbation, dist, exact, times


# test global robustness of f with given delta(s) and epsilon(s) within given radius of the origin
# uses the L-infinity norm for distance
# delta and epsilon may be lists
# bound defines the L-infinity radius about the origin to consider (default 10)
# n_workers specifies the number of processes to assign the (delta, epsilon) cases to (default 1)
# timeout specifies timeout in seconds for each (delta, epsilon) case
# if delta and epsilon are scalars, returns robust (true/false), counterexample, time
# otherwise returns list of (delta, epsilon, robust (true/false), counterexample, time)
# counterexamples are a pair of points in a numpy array with two rows
def global_robustness(f, delta, epsilon, bound=10, n_workers=1, timeout=None):
    onesample = True
    if isinstance(delta, list) or isinstance(delta, np.ndarray): onesample = False
    else: delta = [delta]
    if isinstance(epsilon, list) or isinstance(epsilon, np.ndarray):  onesample = False
    else: epsilon = [epsilon]
    if (len(delta) > 1 or len(epsilon) > 1) and n_workers > 1:
        with multiprocessing.Pool(n_workers) as p:
            results = p.starmap(global_robustness, \
                                [(f, [d], [e], bound, n_workers, timeout) for d in delta for e in epsilon])
            return [r[0] for r in results]
    g = global_adversarial_robustness_transform(f)
    A = np.eye(g.p)
    A[:f.p,f.p:] = -np.eye(f.p)
    A[f.p:,:f.p] = -np.eye(f.p)
    results = []
    for d in delta:
        for e in epsilon:
            b = d*np.ones(g.p)
            D = VerificationDomain(g.p).ball_linf(np.zeros(g.p), bound).lin(A, b)
            robust = True
            start = time()
            for i in range(f.q):
                a = np.zeros(f.q)
                a[i] = 1
                g.set_bounding_coefficients(a)
                v, info = verify_fdt(g, D, b=e, timeout=timeout)
                if v is None or (timeout and time() - start > timeout): 
                    results.append((d, e, None, None, time() - start))
                    robust = False
                    break
                if not v: 
                    results.append((d, e, False, info["x"].reshape(2,-1), time() - start))
                    robust = False
                    break
            if robust: results.append((d, e, True, None, time() - start))
    if onesample: return results[0][2], results[0][3], results[0][4]
    return results
