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
            # Behavior for empty points is debatable:
            # 1. Raise error (as in the prompt's example)
            # 2. Return an empty interval (e.g., Interval.empty(guessed_dim_or_default))
            # For now, let's stick to raising an error if no points are provided.
            # If a dimension can be inferred or a default is desired, this could change.
            raise ValueError("Cannot enclose an empty set of points.")

        if points_arr.ndim == 1:
            # If 1D array, treat as a list of points in 1D space
            # e.g., [1,2,3] -> inf=[1], sup=[3]
            # np.min/max on a 1D array will produce scalars, Interval constructor handles that.
             pass # No reshape needed if we want a 1D interval from a list of scalars.
                     # If points_arr = [p1, p2, p3] (1D array of scalars)
                     # then inf = min(p1,p2,p3), sup = max(p1,p2,p3) which is a 0-dim array
                     # The constructor np.array([val]) will make it 1D.
                     # If it was meant to be points_arr = [[p1x,p1y],[p2x,p2y]] (a 2D array)
                     # and a 1D array like [1,2,3,4] was passed, it's ambiguous.
                     # The matlab version `enclosePoints(points)` takes `points` as matrix where each column is a point.
                     # So `points = [p1_dim1, p2_dim1, ...; p1_dim2, p2_dim2, ...]`
                     # np.min(points, axis=1) would be correct for that interpretation.
                     # Let's assume points are (n_points, dim) as per the prompt.
            points_arr = points_arr.reshape(-1, 1) # Make it (n_points, 1) if it's a flat list.
                                                 # This makes it consistent for np.min/max(axis=0)

        if points_arr.ndim == 0: # Single scalar point
             points_arr = np.array([[points_arr]]) # Make it a (1,1) array

        inf = np.min(points_arr, axis=0)
        sup = np.max(points_arr, axis=0)
        return Interval(inf, sup)

    @staticmethod
    def empty(dim):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("Dimension must be a positive integer.")
        # Represent empty by inf > sup
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
