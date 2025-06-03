import numpy as np
import warnings

# Placeholder for the base class
class ContSet:
    pass

class Zonotope(ContSet):
    def __init__(self, arg1=None, arg2=None):
        if isinstance(arg1, Zonotope): # Copy constructor
            self._c = np.copy(arg1.c) # Use property access to ensure consistency if logic exists there
            self._G = np.copy(arg1.G)
        elif arg1 is not None and arg2 is not None: # Constructor with c and G
            # Use setters to initialize, as they contain validation logic
            self.c = arg1 # This will call the c.setter
            self.G = arg2 # This will call the G.setter
        elif arg1 is not None: # Constructor with combined matrix Z = [c, G] or just c
            # Use Z.setter for this logic
            self.Z = arg1 # This will call the Z.setter
        else:
            # Default constructor: creates a 0-dimensional zonotope (empty center, empty generators)
            self._c = np.array([])
            self._G = np.zeros((0,0))
            # raise ValueError("Invalid arguments for Zonotope constructor. At least one argument is required.")

        # Precedence value, as in Matlab version
        self.precedence = 110

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        c_vec = np.asarray(value)

        if c_vec.ndim == 0: # Handle scalar input, make it 1D array
            c_vec = np.array([c_vec.item()])
        elif c_vec.ndim > 1:
             # Allow column or row vector by flattening
            c_vec = c_vec.flatten()
            # raise ValueError("Center 'c' must be a vector (1D array or flattenable to 1D).")

        # If G exists and is not empty, check for dimension consistency
        if hasattr(self, '_G') and self._G.size > 0 and self._G.shape[0] != c_vec.shape[0]:
            raise ValueError(f"Dimension mismatch: new center has dimension {c_vec.shape[0]}, existing generators have {self._G.shape[0]} rows.")

        self._c = c_vec
        # If G doesn't exist or is empty, and c is not empty, initialize G to match c's dimension with 0 generators
        if c_vec.shape[0] > 0 and (not hasattr(self, '_G') or self._G.size == 0 or self._G.shape[0] != c_vec.shape[0]):
            self._G = np.zeros((c_vec.shape[0], 0))
        elif c_vec.shape[0] == 0 and (not hasattr(self, '_G') or self._G.size == 0): # c is empty, G should be 0xN or 0x0
            if hasattr(self, '_G') and self._G.ndim ==2: # G exists
                 self._G = np.zeros((0, self._G.shape[1]))
            else: # G does not exist or not 2D
                 self._G = np.zeros((0,0))


    @property
    def G(self):
        return self._G

    @G.setter
    def G(self, value):
        G_mat = np.asarray(value)

        # Ensure self.c is initialized.
        if not hasattr(self, '_c'):
            # This implies G is set before c. Let c be initialized based on G's rows, or 0-dim if G is also 0-row/empty.
            if G_mat.ndim == 2 and G_mat.shape[0] > 0 :
                 self._c = np.zeros(G_mat.shape[0]) # Initialize c to match G's dimension
            else: # G is scalar, 1D, or 0-row 2D
                 self._c = np.array([]) # Default to 0-dim c

        dim_c = self._c.shape[0]

        if G_mat.ndim == 0: # Scalar G
            if G_mat.size == 0: # e.g. np.array(None) -> array(None, dtype=object) -> size 1 but tricky
                 G_mat = np.zeros((dim_c, 0)) if dim_c > 0 else np.zeros((0,0))
            else: # Actual scalar value
                item = G_mat.item() if hasattr(G_mat, 'item') else G_mat # handle non-numpy scalar
                if dim_c > 0:
                    G_mat = np.full((dim_c, 1), item)
                else: # dim_c is 0, G should be 0xN
                    G_mat = np.zeros((0,1)) # Or 0x0 if scalar G implies no actual generator an empty space

        elif G_mat.ndim == 1: # Vector G
            if dim_c > 0:
                if G_mat.shape[0] == dim_c: # Matches dimension of c
                    G_mat = G_mat.reshape(dim_c, 1) # Treat as a single generator
                elif G_mat.size == 0: # Empty vector G (e.g. np.array([])) for non-empty c
                     G_mat = np.zeros((dim_c, 0))
                else: # Dimension mismatch
                    raise ValueError(f"Single generator vector G (len {G_mat.shape[0]}) must have dimension {dim_c} or be empty.")
            elif dim_c == 0: # c is empty (0-dim)
                if G_mat.size == 0:
                    G_mat = np.zeros((0,0))
                else: # G has elements but c is 0-dim; G should be 0xN
                    G_mat = np.zeros((0, G_mat.size))

        # Post-processing G_mat for 2D shape if c is defined
        if dim_c > 0 and G_mat.size == 0 and G_mat.shape != (dim_c, 0) : # Ensure empty G is (dim_c, 0)
            G_mat = np.zeros((dim_c, 0))
        elif dim_c == 0 and G_mat.size == 0 and G_mat.shape[0] != 0: # Ensure empty G for empty c is (0, M) or (0,0)
            G_mat = np.zeros((0, G_mat.shape[1] if G_mat.ndim==2 else 0 ))


        # Final check for consistent dimensions if G is not effectively empty for c
        if G_mat.size > 0 and dim_c != G_mat.shape[0]:
            # This case should ideally be caught by earlier logic or imply c needs resizing.
            # However, if c was empty and G has rows, c should have been resized by G.setter's start.
            raise ValueError(f"Dimension mismatch: center has dimension {dim_c}, new generators have {G_mat.shape[0]} rows.")

        self._G = G_mat


    @property
    def Z(self):
        warnings.warn("Property 'Z' is deprecated. Use 'c' and 'G' instead.", DeprecationWarning, stacklevel=2)
        current_c = self.c
        current_G = self.G

        dim_c = current_c.shape[0]

        if dim_c == 0:
            num_g_cols = current_G.shape[1] if current_G.ndim == 2 and current_G.size > 0 else 0
            return np.zeros((0, 1 + num_g_cols ))

        c_col = current_c.reshape(dim_c, 1)

        if current_G.size == 0 or current_G.shape[1] == 0: # No generators or G is (dim,0)
            return c_col

        # Ensure G is compatible if c exists
        if current_G.shape[0] != dim_c:
            # This state should ideally not be reached if setters are correct
            raise ValueError(f"Inconsistent state: c is dim {dim_c}, G has {current_G.shape[0]} rows")

        return np.hstack((c_col, current_G))

    @Z.setter
    def Z(self, value):
        warnings.warn("Property 'Z' is deprecated. Use 'c' and 'G' instead.", DeprecationWarning, stacklevel=2)
        Z_mat = np.asarray(value)

        if Z_mat.ndim == 0 : # Scalar Z treated as [[Z]]
            # This implies a 1-dim zonotope with Z as center and no generators
            self.c = np.array([Z_mat.item()])
            self.G = np.zeros((1,0)) # G.setter will handle this via self.c's new dim
            return

        if Z_mat.ndim == 1: # Vector Z treated as a column vector (center)
            self.c = Z_mat # c.setter will flatten if needed
            self.G = np.zeros((self.c.shape[0],0)) # G.setter will handle this
            return

        # Z_mat is 2D
        if Z_mat.shape[1] == 0: # Empty matrix Z (e.g. shape (N,0)) -> N-dim center, 0 generators
             self.c = np.zeros(Z_mat.shape[0])
             self.G = np.zeros((Z_mat.shape[0],0))
        else:
            self.c = Z_mat[:, 0] # This will trigger c.setter
            # G.setter must be called after c.setter has established the dimension
            self.G = Z_mat[:, 1:] # This will trigger G.setter

    def __repr__(self):
        # Using self.c and self.G to ensure properties are accessed and repr is based on current state
        c_repr = repr(self.c)
        G_mat = self.G

        # Make G representation multi-line if G is multi-line
        G_lines = repr(G_mat).split('\n')
        G_repr = G_lines[0]
        if len(G_lines) > 1:
            # Adjust indent for subsequent lines of G's repr
            indent = ' ' * (len("Zonotope(c=..., G=") -1 + G_lines[0].find('array(') + len('array(') if 'array(' in G_lines[0] else len("       "))
            G_repr += '\n' + '\n'.join([indent + line.lstrip() for line in G_lines[1:]])

        return f"Zonotope(c={c_repr}, G={G_repr})"

    @staticmethod
    def generate_random(dim, num_generators=5, max_center_val=10, max_gen_val=5):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("Dimension 'dim' must be a positive integer.")
        if not isinstance(num_generators, int) or num_generators < 0:
            raise ValueError("Number of generators must be a non-negative integer.")
        if not isinstance(max_center_val, (int, float)) or not isinstance(max_gen_val, (int, float)):
             raise ValueError("max_center_val and max_gen_val must be numbers.")

        c = np.random.uniform(-np.abs(max_center_val), np.abs(max_center_val), dim)
        G = np.random.uniform(-np.abs(max_gen_val), np.abs(max_gen_val), (dim, num_generators))
        return Zonotope(c, G)

    @staticmethod
    def enclose_points(points):
        points_arr = np.asarray(points)

        if points_arr.size == 0:
            # What to return for empty points? An empty 0-dim zonotope? Or error?
            # Or an empty zonotope of a specified dimension if possible?
            # For now, let's raise an error or return a truly empty zonotope.
            # The Matlab version seems to return an empty zonotope obj if points is empty.
            # Zonotope.empty(0) or Zonotope() would give a 0-dim empty zonotope.
            # If points_arr is shape (0,N), it means 0 points in N-dim space.
            if points_arr.ndim == 2 and points_arr.shape[0] == 0 : # (0, dim) array
                dim = points_arr.shape[1]
                return Zonotope.empty(dim)
            raise ValueError("Points array is empty or has an unsupported shape for enclosure.")


        if points_arr.ndim == 1: # Single point (as a 1D array) or a list of scalars for 1D zonotope
            # If it's a single point [x,y,z], reshape to (1, dim)
            # If it's [p1,p2,p3] for a 1D zonotope, needs to be (N,1)
            # Assuming (n_points, dim) structure, a 1D array implies a single point
            points_arr = points_arr.reshape(1, -1)

        if points_arr.ndim != 2 or points_arr.shape[0] == 0: # Should be (n_points, dim)
            raise ValueError("Points must be a 2D array with at least one point (row).")

        dim = points_arr.shape[1]
        if dim == 0: # Points are like [ [], [], [] ] -> (N,0) array
            return Zonotope(np.array([]), np.zeros((0,0))) # 0-dim zonotope

        if points_arr.shape[0] == 1: # Single point, zonotope is just the point
            return Zonotope(points_arr[0,:], np.zeros((dim,0)))

        # For multiple points, create an axis-aligned bounding zonotope
        min_coords = np.min(points_arr, axis=0)
        max_coords = np.max(points_arr, axis=0)

        new_c = (min_coords + max_coords) / 2

        # Generators are half the lengths of the bounding box dimensions, placed along axes
        half_lengths = (max_coords - min_coords) / 2

        # Create diagonal matrix only for non-zero half_lengths to avoid tiny/zero generators
        # G_diag = half_lengths[half_lengths > 1e-9] # Tolerance for "zero"
        # new_G = np.diag(G_diag)
        # if new_G.shape[0] != dim : # Some dimensions might have been filtered
            # Need to reconstruct G with correct dimensions, possibly sparse or with explicit zero columns
            # A simpler way is to create the full diagonal matrix and then filter columns

        temp_G = np.diag(half_lengths)

        # Filter out generator columns that are all zero (or very close to zero)
        # This happens if the point cloud has zero width along an axis
        non_zero_cols_mask = ~np.all(np.isclose(temp_G, 0), axis=0)
        new_G = temp_G[:, non_zero_cols_mask]

        # If all generators are zero (e.g. all points are identical), G becomes (dim, 0)
        if new_G.size == 0 and dim > 0:
            new_G = np.zeros((dim,0))
        elif new_G.size == 0 and dim == 0: # Should be handled by dim==0 case earlier
             new_G = np.zeros((0,0))


        return Zonotope(new_c, new_G)


    @staticmethod
    def empty(dim):
        if not isinstance(dim, int) or dim < 0:
            raise ValueError("Dimension 'dim' must be a non-negative integer.")
        c = np.zeros(dim)
        G = np.zeros((dim, 0)) # Generators are (dim x 0)
        return Zonotope(c, G)

    @staticmethod
    def origin(dim):
        if not isinstance(dim, int) or dim < 0:
            raise ValueError("Dimension 'dim' must be a non-negative integer.")
        # For a zonotope, origin is a point zonotope: center at origin, no generators.
        # This is effectively the same as an "empty" zonotope in terms of generator volume.
        return Zonotope.empty(dim)
