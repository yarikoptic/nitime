"""Tests for Granger 'causality' routines."""

import numpy as np
import numpy.testing as npt
from scipy import linalg

from nitime import block_toeplitz

def test_block_toeplitz_solve():
    for order in [2,3,5]:
        for nseries in [2,4,6]:
            nrow = order*nseries
            # Generate all r's as a single matrix (we 'flatten' the block
            # structure)
            all_r_shape = (nseries, (order+1)*nseries)
            n_elems = np.prod(all_r_shape)
            all_r = np.arange(n_elems, dtype=float).reshape(all_r_shape)

            r_vec = all_r[:,nseries:].T
            r_mat_blocks = all_r[:,:-nseries]
            r_mat = block_toeplitz(r_mat_blocks, nseries)
            a_params = linalg.solve(r_mat, -r_vec)
            # Verify the quality of the linear solve
            yield npt.assert_almost_equal(np.dot(r_mat, a_params ),
                                          -r_vec, 12)
