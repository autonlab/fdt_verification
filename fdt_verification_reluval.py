"""
File: fdt_verification_reluval.py
Author(s): Jack Good
Created: Tue Jan 17 18:32:27 EDT 2023
Description: Verification of fuzzy decision trees using our abstraction with the refinement strategy from ReluVal.
Acknowledgements:
Copyright (c) 2021 Carnegie Mellon University
This code is subject to the license terms contained in the code repo.
"""

import numpy as np
from fdt_verification import FDTNode, linear_split

# get lower and upper bounds for hyperrectangle domain D
def domain_to_interval(D):
    lb = [None]*D.p
    ub = [None]*D.p
    for (a, b) in zip(D.A, D.b):
        i = np.where(a != 0)[0][0]
        if a[i] > 0: ub[i] = b/a[i] if ub[i] is None else min(ub[i], b/a[i])
        if a[i] < 0: lb[i] = b/a[i] if lb[i] is None else max(lb[i], b/a[i])
    return lb, ub


# an inheritor of th FDT class that overrides several functions
# to use the refinement strategy from ReluVal as an experimental baseline
class FDTNodeReluVal(FDTNode):
    
    # initialize from another FDTNode
    def __init__(self, other):
        if other.is_leaf:
            self.is_leaf = True
            self.v = other.v
            self.a = other.a
            self.av = other.av
        else:
            self.is_leaf = False
            self.a = other.a
            self.b = other.b
            self.s = other.s
            self.t = other.t
            self.l = FDTNodeReluVal(other.l)
            self.r = FDTNodeReluVal(other.r)
            self.eps = other.eps
            self.p = other.p
        self.q = other.q
        self.memo = other.memo
        
    # initialize as an internal node
    # see FDTNode.internal
    def internal(*args):
        return FDTNodeReluVal(FDTNode.internal(*args))

    # initialize as a leaf
    # see FDTNode.leaf
    def leaf(*args):
        return FDTNodeReluVal(FDTNode.leaf(*args))
    
    # initialize as a leaf with uniform value
    # see FDTNode.const
    def const(*args):
        return FDTNodeReluVal(FDTNode.const(*args))


    # compute bounds of a^T f(x) on domain D, where f is this FDTNodeReluVal, and bounds on its gradient
    # top is a flag used internally to manage memoization
    # returns:
    #   l the lower bound
    #   u the upper bound
    #   gl the gradient lower bound
    #   gu the gradient upper bound
    #   g the dynamically pruned copy of f
    # see the paper for details on split value v
    def bound_with_grad(self, D, top=True):
        if top: self.forget()
            
        # check for memoization
        if self.memo is not None:
            return self.memo
            
        # leaves just return the value
        if self.is_leaf:
            self.memo = (self.av, self.av, np.zeros(D.p), np.zeros(D.p), self)
            return self.memo
        
        # make sure the split is linear
        if not (self.s == linear_split):
            raise ValueError("for ReluVal-based verification, only the simple linear splitting function is supported")
        
        # compute bounds (see the paper for details)
        smin = self.s(-D.maximize(-self.a) + self.b)
        smax = self.s( D.maximize( self.a) + self.b)
        sbar = 0.5*(smin + smax)
        if self.eps and smax < self.eps: # check if we can remove right subtree
            self.memo = self.l.bound_with_grad(D, False)
            return self.memo
        if self.eps and 1-smin < self.eps:  # check if we can remove left subtree
            self.memo = self.r.bound_with_grad(D, False)
            return self.memo
        lL, uL, glL, guL, fL = self.l.bound_with_grad(D, False)
        lR, uR, glR, guR, fR = self.r.bound_with_grad(D, False)
        l0 = (1-smin)*lL + smin*lR
        l1 = (1-smax)*lL + smax*lR
        u0 = (1-smin)*uL + smin*uR
        u1 = (1-smax)*uL + smax*uR
        l = min(l0, l1)
        u = max(u0, u1)
        
        # compute gradient bounds
        g1l0 = (1-smin)*glL + smin*glR
        g1l1 = (1-smax)*glL + smax*glR
        g1u0 = (1-smin)*guL + smin*guR
        g1u1 = (1-smax)*guL + smax*guR
        g1l = np.minimum(g1l0, g1l1)
        g1u = np.minimum(g1u0, g1u1)
        
        sp1 = np.zeros(D.p) if smin == 0 or smax == 1 else self.a
        sp2 = self.a # other case is caught by dynamic pruning
        g2l0 = sp1*(lR - uL)
        g2l1 = sp2*(lR - uL)
        g2u0 = sp1*(uR - lL)
        g2u1 = sp2*(uR - lL)
        g2l = np.min(np.stack((g2l0, g2l1, g2u0, g2u1)), axis=0)
        g2u = np.max(np.stack((g2l0, g2l1, g2u0, g2u1)), axis=0)
        
        gl = g1l + g2l
        gu = g1u + g2u
        
        # create pruned version of self by using pruned subtrees as children
        g = FDTNodeReluVal.internal(self.a, self.b, self.s, self.t, fL, fR, self.eps)
        
        # memoize results
        self.memo = (l, u, gl, gu, g)
        return self.memo
        
    # compute bounds of a^T f(x) on domain D, where f is this FDTNodeReluVal
    # top is a flag used internally to manage memoization
    # returns:
    #   l the lower bound
    #   u the upper bound
    #   s the best split (v, a, b), where v is the value of the split and a, b are the coefficients and intercept of the split
    #   g the dynamically pruned copy of f
    # see the paper for details on v(f,n)
    def bound(self, D, top=True):
        
        # make sure D is a hyper-interval
        if np.any((D.A != 0).sum(axis=1) > 1): 
            #print(D.A)
            raise ValueError("for ReluVal-based verification, only hyper-interval domains are supported")
            
        l, u, gl, gu, g = self.bound_with_grad(D, True)
        
        # determine split using the strategy from ReluVal
        lb, ub = domain_to_interval(D)
        
        gabs = np.maximum(np.abs(gl), np.abs(gu))
        #print("grad " + str(gabs))
        # this will break if the hyper-interval domain is unbounded
        smear = gabs*(np.array(ub)-np.array(lb))
        i = np.argmax(smear)
        a = np.zeros(D.p)
        a[i] = 1
        b = (lb[i] + ub[i])/2
        s = (smear[i], a, b)
        
        return l, u, s, g

