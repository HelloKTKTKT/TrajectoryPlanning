from __future__ import annotations

from typing import Iterable

import numpy as np

from trajplan.shared_types import FloatArray, Matrix, Vector


def as_float_array(data: float | Iterable[float] | np.ndarray) -> FloatArray:
    """
    Convert input data to a NumPy float64 array.

    This function only guarantees dtype, not shape.
    Shape normalization should be handled by `as_vector` or `as_matrix`.

    Parameters
    ----------
    data
        Input scalar, iterable, or NumPy array.

    Returns
    -------
    FloatArray
        NumPy array with dtype float64.
    """
    return np.asarray(data, dtype=np.float64)


def as_vector(data: float | Iterable[float] | np.ndarray) -> Vector:
    """
    Convert input data to a 1D vector with shape (n,).

    Accepted inputs
    ---------------
    - scalar -> shape (1,)
    - 1D array-like -> shape (n,)
    - 2D row/column vector -> flattened to shape (n,)

    Rejected inputs
    ---------------
    - 2D arrays that are not row/column vectors
    - arrays with ndim > 2

    Parameters
    ----------
    data
        Input data representing a single vector.

    Returns
    -------
    Vector
        1D NumPy array with shape (n,).

    Raises
    ------
    ValueError
        If the input cannot be interpreted as a single vector.
    """
    array = as_float_array(data)

    if array.ndim == 0:
        return array.reshape(1)

    if array.ndim == 1:
        return array

    if array.ndim == 2:
        if 1 in array.shape:
            return array.reshape(-1)
        raise ValueError(
            f"Expected a single vector, but got a 2D array with shape {array.shape}."
        )

    raise ValueError(
        f"Expected at most 2 dimensions for a vector, but got array with ndim={array.ndim}."
    )


def as_vector3(data: float | Iterable[float] | np.ndarray) -> Vector:
    """
    Convert input data to a 3D vector with shape (3,).

    Parameters
    ----------
    data
        Input data representing one 3D vector.

    Returns
    -------
    Vector
        1D NumPy array with shape (3,).

    Raises
    ------
    ValueError
        If the input cannot be interpreted as a 3D vector.
    """
    vector = as_vector(data)

    if vector.shape != (3,):
        raise ValueError(
            f"Expected a 3D vector with shape (3,), but got {vector.shape}."
        )

    return vector


def as_matrix(data: Iterable[float] | Iterable[Iterable[float]] | np.ndarray) -> Matrix:
    """
    Convert input data to a 2D matrix with shape (N, n).

    Accepted inputs
    ---------------
    - 1D array-like -> interpreted as a single row, shape becomes (1, n)
    - 2D array-like -> kept as shape (N, n)

    Rejected inputs
    ---------------
    - scalar
    - arrays with ndim > 2

    Parameters
    ----------
    data
        Input data representing stacked vectors or a matrix.

    Returns
    -------
    Matrix
        2D NumPy array with shape (N, n).

    Raises
    ------
    ValueError
        If the input cannot be interpreted as a matrix.
    """
    array = as_float_array(data)

    if array.ndim == 0:
        raise ValueError("A scalar cannot be converted to a 2D matrix.")

    if array.ndim == 1:
        return array.reshape(1, -1)

    if array.ndim == 2:
        return array

    raise ValueError(
        f"Expected at most 2 dimensions for a matrix, but got array with ndim={array.ndim}."
    )


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a scalar value into the interval [min_value, max_value].

    Parameters
    ----------
    value
        Input scalar.
    min_value
        Lower bound.
    max_value
        Upper bound.

    Returns
    -------
    float
        Clamped scalar value.

    Raises
    ------
    ValueError
        If min_value is greater than max_value.
    """
    if min_value > max_value:
        raise ValueError(
            f"Expected min_value <= max_value, but got {min_value} > {max_value}."
        )

    return max(min_value, min(value, max_value))
