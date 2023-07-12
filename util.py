import scipy
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

# ------------------------------------------------------------------------------------------------------

def centering_matrix(n):
    return np.identity(n) - np.full((n, n), 1 / n)

# ------------------------------------------------------------------------------------------------------

def row_centre(A, B=None):
    B = B if B is not None else A
    row_means = np.mean(B, axis=1, keepdims=True)
    return A - row_means

# ------------------------------------------------------------------------------------------------------

def col_centre(A, B=None):
    B = B if B is not None else A
    col_means = np.mean(B, axis=0, keepdims=True)
    return A - col_means

# ------------------------------------------------------------------------------------------------------

def col_row_centre(A, row_mat=None, col_mat=None):
    return col_centre(row_centre(A, row_mat), col_mat)

# ------------------------------------------------------------------------------------------------------

def row_col_centre(A, row_mat=None, col_mat=None):
    return row_centre(col_centre(A, col_mat), row_mat)

# ------------------------------------------------------------------------------------------------------

def centre_kernel(K, centre_data):
    return row_col_centre(K) if centre_data else col_centre(K)

# ------------------------------------------------------------------------------------------------------

def covar_matrix(K, centre_data):
    J = centre_kernel(K, centre_data)
    return J.T @ J

# ------------------------------------------------------------------------------------------------------

def rayleigh_numerator(X, centre_data, kernel, **kernel_args):
    K = pairwise_kernels(X, metric=kernel, **kernel_args)
    return covar_matrix(K, centre_data)

# ------------------------------------------------------------------------------------------------------

def rayleigh_denominator(X, y, centre_data, regularization, reg_param, kernel, **kernel_args):
    
    K = pairwise_kernels(X, metric=kernel, **kernel_args)

    rayleigh_denom = np.zeros(K.shape)
    for label in np.unique(y):
        rayleigh_denom += covar_matrix(K[y == label], centre_data)

    I = np.identity(K.shape[0])
    rayleigh_denom += 1e-6 * I                                                   # ensure positive-definiteness
    rayleigh_denom += reg_param * (I if regularization == 'coefficients' else K) # regularization

    return rayleigh_denom

# ------------------------------------------------------------------------------------------------------

def p_matrix(C):
    if C.ndim == 1:
        P = np.identity(C.shape[0]) - (np.outer(C, C) / np.inner(C, C))
    else:
        P = np.identity(C.shape[1]) - (C.T @ np.linalg.pinv(C @ C.T, hermitian=True) @ C)
    return P

# ------------------------------------------------------------------------------------------------------

def sqrt_inv(A):
    U, S, VT = np.linalg.svd(A)
    return U @ np.diag(1 / np.sqrt(S)) @ VT

# ------------------------------------------------------------------------------------------------------

def leading_eigs(A, B=None):
    eig_vals, eig_vecs = scipy.linalg.eigh(A, B)
    return eig_vals[-1], eig_vecs[:, -1]

# ------------------------------------------------------------------------------------------------------

def optimization(X, y, output_dim, centre_data, regularization, reg_param, kernel, **kernel_args):
    
    A = rayleigh_numerator(X, centre_data, kernel, **kernel_args)
    B = rayleigh_denominator(X, y, centre_data, regularization, reg_param, kernel, **kernel_args)
    
    val, vec = leading_eigs(A, B)
    quotients = [val]
    coefficients = vec
    
    for i in range(1, output_dim):
        D = sqrt_inv(B)
        P = p_matrix(coefficients @ A @ D)
        val, vec = leading_eigs(P @ (D @ A @ D) @ P)
        
        quotients.append(val)
        coefficients = np.vstack((coefficients, D @ vec))

    return quotients, coefficients