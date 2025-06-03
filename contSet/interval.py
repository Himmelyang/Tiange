import numpy as np

# Placeholder for the base class
class ContSet:
    pass

class Interval(ContSet):
    def __init__(self, arg1=None, arg2=None):
        if isinstance(arg1, Interval):
            # Copy constructor
            self.inf = np.copy(arg1.inf)
            self.sup = np.copy(arg1.sup)
        elif arg1 is not None and arg2 is not None:
            # Constructor with lower and upper bounds
            inf_val = np.asarray(arg1)
            sup_val = np.asarray(arg2)

            # Ensure inf_val and sup_val are at least 1-D if they are single numbers
            if inf_val.ndim == 0:
                inf_val = np.array([inf_val])
            if sup_val.ndim == 0:
                sup_val = np.array([sup_val])

            if inf_val.shape != sup_val.shape:
                raise ValueError("Lower and upper bounds must have the same shape.")

            if not np.all(inf_val <= sup_val):
                # Allow for a small tolerance, similar to Matlab version
                if not np.all(inf_val < sup_val + 1e-7): # Adjusted tolerance check
                    raise ValueError("Lower bound must be less than or equal to upper bound for all elements.")

            self.inf = inf_val
            self.sup = sup_val
        elif arg1 is not None:
            # Constructor with a single value (inf = sup = value)
            val = np.asarray(arg1)
            if val.ndim == 0: # Ensure it's an array
                val = np.array([val])
            self.inf = val
            self.sup = val
        else:
            raise ValueError("Invalid arguments for Interval constructor. Must provide at least one argument.")

        # Placeholder for precedence, as in Matlab version
        self.precedence = 120

    def __repr__(self):
        return f"Interval(inf={self.inf}, sup={self.sup})"

    @property
    def dim(self):
        # Dimension is the length of the inf/sup vectors
        # For a 0-element array (e.g. np.array([]) which has shape (0,)), dim is 0.
        # For a scalar array (e.g. np.array([5]) which has shape (1,)), dim is 1.
        if self.inf.ndim == 0 : # Should not happen if constructor enforces array
            return 0
        return self.inf.shape[0]

    def is_empty(self):
        # An interval is empty if any inf > sup.
        # Based on constructor logic, this means it was created with Interval.empty()
        # or resulted from an operation that produces an empty interval.
        if self.inf.size == 0 and self.sup.size == 0: # 0-dim interval (shape (0,))
            # Conventionally, a 0-dimensional interval might be considered not empty, or its emptiness
            # is not well-defined by inf > sup if there are no elements to compare.
            # Let's define it as not empty for consistency (np.any([]) is False).
            return False
        return np.any(self.inf > self.sup)

    def __add__(self, other):
        if self.is_empty():
            # If self is empty, the result is an empty interval of the same dimension.
            return Interval.empty(self.dim)

        if isinstance(other, Interval):
            if other.is_empty():
                # If other is empty, result is an empty interval.
                # Dimension should be compatible; use self's dim or other's if self is 0-dim.
                return Interval.empty(self.dim if self.dim > 0 else other.dim)

            # Dimension compatibility check for Interval + Interval
            # Both must have same dimension, or one must be scalar-like (dim 1, size 1)
            # to be broadcast with the other.

            s_dim = self.dim
            o_dim = other.dim
            s_size = self.inf.size
            o_size = other.inf.size

            if s_dim == o_dim:
                new_inf = self.inf + other.inf
                new_sup = self.sup + other.sup
            elif s_dim == 1 and s_size == 1: # self is scalar-like e.g. Interval([0],[1])
                new_inf = self.inf[0] + other.inf
                new_sup = self.sup[0] + other.sup
            elif o_dim == 1 and o_size == 1: # other is scalar-like
                new_inf = self.inf + other.inf[0]
                new_sup = self.sup + other.sup[0]
            else:
                raise ValueError(f"Dimension mismatch for Interval addition: self dim {s_dim} (size {s_size}), other dim {o_dim} (size {o_size})")

            return Interval(new_inf, new_sup)

        elif isinstance(other, (int, float, np.number, np.ndarray)):
            other_arr = np.asarray(other)

            # Handle Interval + numeric
            # If self is scalar-like: Interval([s_inf], [s_sup]) + arr -> Interval(s_inf + arr, s_sup + arr)
            # If other_arr is scalar: Interval(inf, sup) + scalar -> Interval(inf + scalar, sup + scalar)
            # If self.inf and other_arr have compatible shapes for broadcasting:

            if self.dim == 1 and self.inf.size == 1 and (other_arr.ndim > 0 and other_arr.size > 1):
                # Self is scalar-like, other is a non-scalar array
                new_inf = self.inf[0] + other_arr
                new_sup = self.sup[0] + other_arr
            elif other_arr.ndim == 0 or other_arr.size == 1:
                # Other is a scalar or scalar-like array
                new_inf = self.inf + other_arr # Broadcasting handles other_arr being scalar
                new_sup = self.sup + other_arr
            elif self.inf.shape == other_arr.shape :
                 # Both are non-scalar arrays of the same shape
                new_inf = self.inf + other_arr
                new_sup = self.sup + other_arr
            elif self.dim == 0 and self.inf.size == 0: # self is Interval([],[]) or Interval.empty(0)
                # Adding numeric to a 0-dim interval results in a new interval of that numeric's shape
                # e.g. Interval.empty(0) + 5 -> Interval([5],[5])
                # e.g. Interval.empty(0) + [1,2] -> Interval([1,2],[1,2])
                new_inf = np.array([]) + other_arr # relies on numpy's behavior for [] + arr
                new_sup = np.array([]) + other_arr
            else:
                raise ValueError(f"Dimension mismatch or incompatible shapes for Interval + numeric: self.inf shape {self.inf.shape}, other shape {other_arr.shape}")

            return Interval(new_inf, new_sup)

        else:
            # Precedence check for other ContSet types would go here.
            # For now, if not Interval or numeric, it's not implemented.
            return NotImplemented

    def __radd__(self, other):
        # This handles cases like `numeric + Interval`
        # We can just call __add__ as it's equipped to handle numeric types for the 'other' argument.
        if isinstance(other, (int, float, np.number, np.ndarray)):
            return self.__add__(other)
        else:
            return NotImplemented

    @staticmethod
    def generate_random(dim, min_val=-10, max_val=10, max_width=5):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("Dimension must be a positive integer.")
        if max_width < 0:
            raise ValueError("Maximum width must be non-negative.")
        # Generate random center points
        center = np.random.uniform(min_val, max_val, dim)
        # Generate random widths (can be zero, up to max_width)
        width = np.random.uniform(0, max_width, dim)
        inf = center - width / 2
        sup = center + width / 2
        return Interval(inf, sup)

    @staticmethod
    def enclose_points(points):
        points_arr = np.asarray(points)

        if points_arr.size == 0:
            raise ValueError("Cannot enclose an empty set of points.")

        if points_arr.ndim == 1:
            points_arr = points_arr.reshape(-1, 1)

        if points_arr.ndim == 0:
             points_arr = np.array([[points_arr]])

        inf = np.min(points_arr, axis=0)
        sup = np.max(points_arr, axis=0)
        return Interval(inf, sup)

    @staticmethod
    def empty(dim):
        if not isinstance(dim, int) or dim < 0: # Allow dim=0 for empty interval
            raise ValueError("Dimension must be a non-negative integer.")
        if dim == 0:
            return Interval(np.array([]), np.array([])) # inf=[], sup=[] for 0-dim
        # Represent non-0-dim empty by inf > sup
        return Interval(np.ones(dim), np.zeros(dim))

    @staticmethod
    def full_space(dim):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("Dimension must be a positive integer.")
        return Interval(np.full(dim, -np.inf), np.full(dim, np.inf))

    @staticmethod
    def origin(dim):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("Dimension must be a positive integer.")
        return Interval(np.zeros(dim), np.zeros(dim))
