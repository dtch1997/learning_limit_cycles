import abc
from jax import grad

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
    def __init__(self, s1, s2, a1 = 0.5, a2 = 0.5):
        self.s1 = s1
        self.s2 = s2
        self.a1 = a1
        self.a2 = a2 
    def get_gradient(self, x):
        return self.a1 * self.s1.get_gradient(x) \
            + self.a2 * self.s2.get_gradient(x)

class LinearCombinationPotentialField(LinearCombinationVectorField):
    def get_value(self, x):
        return self.a1 * self.s1.get_value(x) \
            + self.a2 * self.s2.get_value(x)

        