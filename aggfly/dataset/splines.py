"""Fixed spline bases for lazy climate transforms (no fitting to observations)."""

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import null_space


def validate_spline(degree=1, knots=(20,), restricted=False, basis="truncated_power"):
    """Validate a specification shared by the Python API and CLI."""
    if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer)) or degree not in range(1, 5):
        raise ValueError("spline degree must be an integer from 1 to 4")
    if not isinstance(restricted, (bool, np.bool_)):
        raise ValueError("spline restricted must be a boolean")
    if basis not in ("truncated_power", "bspline"):
        raise ValueError("spline basis must be 'truncated_power' or 'bspline'")
    try:
        k = np.asarray(knots, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("spline knots must be finite, strictly increasing numbers") from None
    if k.ndim != 1 or not len(k) or not np.isfinite(k).all() or not (np.diff(k) > 0).all():
        raise ValueError("spline knots must be finite, strictly increasing numbers")
    if restricted and len(k) < max(2, degree):
        raise ValueError("restricted splines require at least max(2, degree) distinct knots")
    return k


def spline_size(degree=1, knots=(20,), restricted=False, basis="truncated_power"):
    k = validate_spline(degree, knots, restricted, basis)
    return len(k) - degree + 2 if restricted else len(k) + degree


def _evaluate_bspline(x, spline, bounds):
    # SciPy evaluates eager NumPy blocks; Dask retains the original chunk layout.
    x = np.asarray(x, dtype=float)
    result = spline(x)
    if bounds is not None:
        # Use points inside the outer pieces: at a degree-1 knot the
        # derivative is discontinuous, so the side used matters.
        left, right = bounds
        result = np.where(x < left, spline(left) + (x-left)*spline(left, nu=1), result)
        result = np.where(x > right, spline(right) + (x-right)*spline(right, nu=1), result)
    return result


def spline_arrays(array, degree=1, knots=(20,), restricted=False, basis="truncated_power"):
    """Return (labels, DataArrays), omitting a constant/intercept column.

    Restricted degree p preserves C^(p-1) continuity and has linear tails.
    B-spline and power bases span the same space *including a constant*;
    their omitted-intercept conventions differ. For period sums, retain a
    day-count regressor when exposure duration varies between observations.
    """
    k = validate_spline(degree, knots, restricted, basis)
    x = array.astype(float)
    if basis == "truncated_power":
        powers = [1] if restricted else range(1, degree + 1)
        labels = [f"power_{p}" for p in powers]
        arrays = [x ** p for p in powers]
        n = len(k) - degree + 1 if restricted else len(k)
        for j in range(n):
            term = (x - k[j]).clip(min=0) ** degree
            if restricted and degree > 1:
                # Lagrange weights cancel powers T^p through T^2. For p=3
                # this is exactly stagg's unnormalized restricted cubic basis.
                tail = k[-(degree - 1):]
                for q, knot in enumerate(tail):
                    others = np.delete(tail, q)
                    coef = np.prod((k[j] - others) / (knot - others))
                    term = term - coef * (x - knot).clip(min=0) ** degree
            labels.append(f"term_{j + 1}")
            arrays.append(term)
        return labels, arrays

    # Auxiliary bounds enclose all user knots, which are breakpoints. They
    # introduce no extra breakpoints: extrapolation continues the end pieces.
    span = max(float(k[-1] - k[0]), 1.0)
    left, right = k[0] - span, k[-1] + span
    t = np.r_[np.repeat(left, degree + 1), k, np.repeat(right, degree + 1)]
    count = len(t) - degree - 1
    raw = BSpline(t, np.eye(count), degree, extrapolate=True)
    constraints = []
    if restricted:
        # Setting derivatives 2..p to zero on each outer polynomial forces
        # the entire outer piece to be linear, with standard knot smoothness.
        for endpoint in (left, right):
            for order in range(2, degree + 1):
                row = raw(endpoint, nu=order)
                constraints.append(row / np.linalg.norm(row))
    # Fix the otherwise arbitrary constant by anchoring the basis at left.
    constraints.append(raw(left))
    coefficients = null_space(np.asarray(constraints))
    labels = [f"bspline_{j + 1}" for j in range(coefficients.shape[1])]
    arrays = [x.copy(data=x.data.map_blocks(
        _evaluate_bspline,
        spline=BSpline(t, coefficients[:, j], degree, extrapolate=True),
        bounds=(left, right) if restricted else None, dtype=float
    )) for j in range(len(labels))]
    return labels, arrays
