from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]

Vector: TypeAlias = FloatArray
Matrix: TypeAlias = FloatArray
IntVector: TypeAlias = NDArray[np.int64]

GridIndex: TypeAlias = tuple[int, int, int]
