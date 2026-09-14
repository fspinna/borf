"""Moving mean and standard deviation, weighted by the time between observations."""

import numba as nb
import numpy as np

FASTMATH = True


@nb.njit(fastmath=FASTMATH, cache=True)
def weighted_moving_average_textbook(arr, weights, window_width):
    out = np.full(arr.shape, np.nan)
    W_sum_prev = np.sum(weights[:window_width])
    arr_times_W_sum_prev = np.sum(weights[:window_width] * arr[:window_width])
    out[window_width - 1] = arr_times_W_sum_prev / W_sum_prev
    for i in range(window_width, len(arr)):
        W_sum_prev = W_sum_prev - weights[i - window_width] + weights[i]
        arr_times_W_sum_prev = (
            arr_times_W_sum_prev
            - arr[i - window_width] * weights[i - window_width]
            + arr[i] * weights[i]
        )
        out[i] = arr_times_W_sum_prev / W_sum_prev
    return out


@nb.njit(fastmath=FASTMATH, cache=True)
def weighted_moving_standard_deviation_welford(arr, weights, window_width):
    out = np.full(arr.shape, np.nan)
    W_sum_prev = np.sum(weights[:window_width])
    arr_times_W_sum_prev = np.sum(weights[:window_width] * arr[:window_width])
    ss_prev = (
        np.sum(weights[:window_width] * (arr[:window_width]) ** 2)
        - arr_times_W_sum_prev**2 / W_sum_prev
    )
    mu_prev = arr_times_W_sum_prev / W_sum_prev
    out[window_width - 1] = ss_prev / W_sum_prev
    for i in range(window_width, len(arr)):
        W_sum_next = W_sum_prev - weights[i - window_width] + weights[i]
        arr_times_W_sum_next = (
            arr_times_W_sum_prev
            - arr[i - window_width] * weights[i - window_width]
            + arr[i] * weights[i]
        )
        mu_next = arr_times_W_sum_next / W_sum_next
        ss_prev = (
            ss_prev
            + weights[i] * (arr[i] - mu_prev) * (arr[i] - mu_next)
            - weights[i - window_width]
            * (arr[i - window_width] - mu_prev)
            * (arr[i - window_width] - mu_next)
        )
        out[i] = ss_prev / W_sum_next
        arr_times_W_sum_prev = arr_times_W_sum_next
        W_sum_prev = W_sum_next
        mu_prev = mu_next
    return np.sqrt(out)
