import abc
import numpy as np
import jax.numpy as jnp

from typing import List
from jax import grad, jacfwd

class VectorField(abc.ABC):
    """Abstract class for a 2D vector field"""
    @abc.abstractmethod
    def get_gradient(self, x):
        """Must be implemented in Jax"""
        pass

class PotentialField(VectorField):
    """Abstract base class for a potential field"""
    @abc.abstractmethod
    def get_value(self, x):
        """Must be implemented in Jax"""
        pass
    def get_gradient_fn(self):
        return grad(self.get_value)
    def get_gradient(self, x):
        return grad(self.get_value)(x)

class FunctionalPotentialField(PotentialField):
    """Potential field parametrized by a function R^2 -> R"""
    def __init__(self, f):
        self.f = f 
    def get_value(self, x):
        return self.f(x)

class LinearCombinationVectorField(VectorField):
    def __init__(self, vs: List[VectorField], cs: List[float] = None):
        assert len(vs) >= 1, "LinearCombinationVectorField must wrap at least 1 base field"
        if cs is None:
            cs = [1 / len(vs)] * len(vs)
        self.cs = cs 
        self.vs = vs

    def get_gradient(self, x):
        return sum([c * v.get_gradient(x) for c, v in zip(self.cs, self.vs)])

class LinearCombinationPotentialField(LinearCombinationVectorField):
    def get_value(self, x):
        return sum([c * v.get_value(x) for c, v in zip(self.cs, self.vs)])

class SmoothTransformationVectorField(VectorField):
    def __init__(self, v: VectorField, f):
        self.v = v 
        self.f = f
    
    def get_gradient(self, x):
        return grad(self.f)(x) @ self.v.get_gradient(x)

class SmoothTransformationVectorField(VectorField):
    """
    v: A vector field. 
    f: A smooth mapping on R^n to R^n, implemented in Jax.
    """
    def __init__(self, v: VectorField, f):
        self.v = v 
        self.f = f
    
    def get_gradient(self, x):

        # Proof that this formula is correct: 
        # 
        # f: M --> N is a smooth map on manifolds
        # Df_x: TM_x --> TN_x is its differential at x
        # v: N --> TN is a vector field on N
        # 
        # We seek the image of v(x) under f. 
        # By homomorphism, this is given by inv(Df_x) * v * f
        return jnp.linalg.inv(jacfwd(self.f)(x)) \
            @ self.v.get_gradient(self.f(x))
    
    
class SmoothTransformationPotentialField(PotentialField):
    def __init__(self, p: PotentialField, f):
        self.p = p 
        self.f = f
    
    def get_value(self, x):
        # For a smooth function f, and potential
        # function p, we provide the potential p(f(x))
            return self.p.get_value(self.f(x))




