import numpy as np

def higuchi_fractal_dimension(X, k_max=10):
    """
    Calculates the Higuchi Fractal Dimension (HFD) of a time series.

    Args:
        X (array-like): The input time series.
        k_max (int): The maximum interval length.

    Returns:
        float: The estimated fractal dimension.
    """
    X = np.asarray(X)
    N = len(X)
    L = np.zeros(k_max)
    for k in range(1, k_max + 1):
        Lk = 0
        for m in range(k):
            n_max = int((N - m - 1) // k)
            if n_max == 0:
                continue

            # Vectorized sum of differences
            indices_i = m + np.arange(1, n_max + 1) * k
            indices_prev = m + np.arange(0, n_max) * k
            sum_diff = np.sum(np.abs(X[indices_i] - X[indices_prev]))

            norm = (N - 1) / (n_max * k)
            Lk += (sum_diff * norm) / k

        L[k-1] = Lk / k

    valid = L > 0
    if np.sum(valid) < 2:
        return 0.0

    log_k = np.log(np.arange(1, k_max + 1)[valid])
    log_L = np.log(L[valid])

    coeffs = np.polyfit(log_k, log_L, 1)
    return -coeffs[0]
