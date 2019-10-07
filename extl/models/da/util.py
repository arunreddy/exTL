"""
Utility functions necessary for different classifiers.

Contains algebraic operations, and label encodings.
"""

import numpy as np
import numpy.linalg as al
import scipy.stats as st


def one_hot(y, fill_k=False, one_not=False):
    """Map to one-hot encoding."""
    # Check labels
    labels = np.unique(y)

    # Number of classes
    K = len(labels)

    # Number of samples
    N = y.shape[0]

    # Preallocate array
    if one_not:
        Y = -np.ones((N, K))
    else:
        Y = np.zeros((N, K))

    # Set k-th column to 1 for n-th sample
    for n in range(N):

        # Map current class to index label
        y_n = (y[n] == labels)

        if fill_k:
            Y[n, y_n] = y_n
        else:
            Y[n, y_n] = 1

    return Y, labels


def regularize_matrix(A, a=0.0):
    """
    Regularize matrix by ensuring minimum eigenvalues.

    INPUT   (1) array 'A': square matrix
            (2) float 'a': constraint on minimum eigenvalue
    OUTPUT  (1) array 'B': constrained matrix
    """
    # Check for square matrix
    N, M = A.shape
    if not N == M:
        raise ValueError('Matrix not square.')

    # Check for valid matrix entries
    if np.any(np.isnan(A)) or np.any(np.isinf(A)):
        raise ValueError('Matrix contains NaNs or infinities.')

    # Check for non-negative minimum eigenvalue
    if a < 0:
        raise ValueError('minimum eigenvalue cannot be negative.')

    elif a == 0:
        return A

    else:
        # Ensure symmetric matrix
        A = (A + A.T) / 2

        # Eigenvalue decomposition
        E, V = al.eig(A)

        # Regularization matrix
        aI = a * np.eye(N)

        # Subtract regularization
        E = np.diag(E) + aI

        # Cap negative eigenvalues at zero
        E = np.maximum(0, E)

        # Reconstruct matrix
        B = np.dot(np.dot(V, E), V.T)

        # Add back subtracted regularization
        return B + aI


def is_pos_def(X):
    """Check for positive definiteness."""
    return np.all(np.linalg.eigvals(X) > 0)


def nullspace(A, atol=1e-13, rtol=0):
    """
    Compute an approximate basis for the nullspace of A.

    INPUT   (1) array 'A': 1-D array with length k will be treated
                as a 2-D with shape (1, k).
            (2) float 'atol': the absolute tolerance for a zero singular value.
                Singular values smaller than `atol` are considered to be zero.
            (3) float 'rtol': relative tolerance. Singular values less than
                rtol*smax are considered to be zero, where smax is the largest
                singular value.

                If both `atol` and `rtol` are positive, the combined tolerance
                is the maximum of the two; tol = max(atol, rtol * smax)
                Singular values smaller than `tol` are considered to be zero.
    OUTPUT  (1) array 'B': if A is an array with shape (m, k), then B will be
                an array with shape (k, n), where n is the estimated dimension
                of the nullspace of A.  The columns of B are a basis for the
                nullspace; each element in np.dot(A, B) will be
                approximately zero.
    """
    # Expand A to a matrix
    A = np.atleast_2d(A)

    # Singular value decomposition
    u, s, vh = al.svd(A)

    # Set tolerance
    tol = max(atol, rtol * s[0])

    # Compute the number of non-zero entries
    nnz = (s >= tol).sum()

    # Conjugate and transpose to ensure real numbers
    ns = vh[nnz:].conj().T

    return ns
