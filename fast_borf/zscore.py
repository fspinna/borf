import numba as nb

from fast_borf.constants import FASTMATH


@nb.njit(fastmath=FASTMATH, cache=True)
def zscore(a: float, mu: float, sigma: float) -> float:
    if sigma == 0:
        return 0
    return (a - mu) / sigma


@nb.njit(fastmath=FASTMATH, cache=True)
def zscore_threshold(
    a: float, mu: float, sigma: float, sigma_global: float, sigma_threshold: float
) -> float:
    if sigma_global == 0:
        return 0
    if sigma / sigma_global < sigma_threshold:
        return 0
    return zscore(a=a, mu=mu, sigma=sigma)
