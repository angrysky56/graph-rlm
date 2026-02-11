"""
Higuchi Fractal Dimension Skill.

Calculates the Higuchi Fractal Dimension (HFD) of a time series.
"""

import numpy as np


def higuchi_fractal_dimension(data, k_max=10):
    """
    Calculates the Higuchi Fractal Dimension (HFD) of a time series.

    Args:
        data (array-like): The input time series.
        k_max (int): The maximum interval length.

    Returns:
        float: The estimated fractal dimension.
    """
    data = np.asarray(data)
    n_samples = len(data)
    fractal_lengths = np.zeros(k_max)

    for k in range(1, k_max + 1):
        length_k = 0.0
        for m in range(k):
            n_max = int((n_samples - m - 1) // k)
            if n_max == 0:
                continue

            # Vectorized sum of differences
            # Create indices for the current subsequence
            # indices_i corresponds to i*k + m
            # We want to sum |X(i*k + m) - X((i-1)*k + m)| for i = 1 to n_max
            # So current index is (arange(1, n_max+1) * k) + m
            # Previous index is (arange(0, n_max) * k) + m
            indices_i = m + np.arange(1, n_max + 1) * k
            indices_prev = m + np.arange(0, n_max) * k

            sum_diff = float(np.sum(np.abs(data[indices_i] - data[indices_prev])))

            # scaling factor for this m
            norm = (n_samples - 1) / (n_max * k)

            # Add to the length for this k
            # The formula typically is L_m(k) = (sum_diff * norm) / k
            # We sum L_m(k) for all m, then divide by k at the end?
            # Or is L(k) = mean(L_m(k))?
            # Standard: L(k) = (1/k) * sum_{m=1}^{k} L_m(k).
            # Here length_k sums (sum_diff * norm) / k.
            # So length_k = sum_{m=1}^{k} L_m(k).
            length_k += (sum_diff * norm) / k

        # Finally L(k) is the average over m, but strictly speaking the logic usually
        # sums them then divides. The implementation divides by k again:
        # L[k-1] = length_k / k
        # If length_k is sum(L_m), then length_k/k is mean(L_m).
        # This matches the definition of L(k) as the average length of the k sub-series.
        fractal_lengths[k - 1] = length_k / k

    # Filter out zeros to avoid log(0)
    valid = fractal_lengths > 0
    if np.sum(valid) < 2:
        return 0.0

    log_k = np.log(np.arange(1, k_max + 1)[valid])
    log_lengths = np.log(fractal_lengths[valid])

    # Slope of log(L(k)) vs log(1/k) is D.
    # Since we plot log(L(k)) vs log(k), the slope is -D.
    # So D = -slope.
    coeffs = np.polyfit(log_k, log_lengths, 1)
    return -coeffs[0]
