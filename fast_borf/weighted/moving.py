import numpy as np
import numba as nb
from fast_borf.constants import FASTMATH


@nb.njit
def weighted_moving_average_naive(arr, weights, window_width):
    out = np.full(arr.shape, np.nan)
    for i in range(window_width-1, len(arr)):
        out[i] = np.sum(arr[i - window_width + 1:i + 1] * weights[i - window_width + 1:i + 1]) / np.sum(
            weights[i - window_width + 1:i + 1]
        )
    return out


@nb.njit
def weighted_running_moving_average_naive(arr, weights, window_width):
    out = np.full(arr.shape, np.nan)

    for i in range(min(window_width, len(arr))):
        W_sum_prev = np.sum(weights[:i + 1])
        arr_times_W_sum_prev = np.sum(arr[:i + 1] * weights[:i + 1])
        out[i] = arr_times_W_sum_prev / W_sum_prev  # Cumulative weighted average

    for i in range(window_width, len(arr)):
        W_sum_prev = np.sum(weights[i - window_width + 1:i + 1])
        arr_times_W_sum_prev = np.sum(arr[i - window_width + 1:i + 1] * weights[i - window_width + 1:i + 1])
        out[i] = arr_times_W_sum_prev / W_sum_prev

    return out


@nb.njit
def weighted_moving_standard_deviation_naive(arr, weights, window_width):
    out = np.full(arr.shape, np.nan)
    for i in range(window_width-1, len(arr)):
        W_sum_prev = np.sum(weights[i - window_width + 1:i + 1])
        arr_times_W_sum_prev = np.sum(arr[i - window_width + 1:i + 1] * weights[i - window_width + 1:i + 1])
        ss_prev = np.sum(weights[i - window_width + 1:i + 1] * (arr[i - window_width + 1:i + 1]) ** 2) - arr_times_W_sum_prev ** 2 / W_sum_prev
        out[i] = np.sqrt(ss_prev / W_sum_prev)
    return out


@nb.njit
def weighted_running_moving_standard_deviation_naive(arr, weights, window_width):
    out = np.full(arr.shape, np.nan)
    out[0] = 0.0
    for i in range(1, min(window_width, len(arr))):
        W_sum_prev = np.sum(weights[:i + 1])
        arr_times_W_sum_prev = np.sum(arr[:i + 1] * weights[:i + 1])
        ss_prev = np.sum(weights[:i + 1] * (arr[:i + 1] ** 2)) - arr_times_W_sum_prev ** 2 / W_sum_prev
        out[i] = np.sqrt(ss_prev / W_sum_prev)
    for i in range(window_width, len(arr)):
        W_sum_prev = np.sum(weights[i - window_width + 1:i + 1])
        arr_times_W_sum_prev = np.sum(arr[i - window_width + 1:i + 1] * weights[i - window_width + 1:i + 1])
        ss_prev = np.sum(weights[i - window_width + 1:i + 1] * (
        arr[i - window_width + 1:i + 1]) ** 2) - arr_times_W_sum_prev ** 2 / W_sum_prev
        out[i] = np.sqrt(ss_prev / W_sum_prev)

    return out


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
def weighted_running_moving_average_textbook(arr, weights, window_width):
    out = np.full(arr.shape, np.nan)

    W_sum_prev = 0.0
    arr_times_W_sum_prev = 0.0

    for i in range(min(window_width, len(arr))):
        W_sum_prev += weights[i]
        arr_times_W_sum_prev += weights[i] * arr[i]
        out[i] = arr_times_W_sum_prev / W_sum_prev

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


@nb.njit(fastmath=FASTMATH, cache=True)
def weighted_running_moving_standard_deviation_welford(arr, weights, window_width):
    out = np.full(arr.shape, np.nan)

    W_sum_prev = 0.0
    arr_times_W_sum_prev = 0.0
    ss_prev = 0.0
    mu_prev = 0.0

    # Compute cumulative weighted standard deviation until window_width is reached
    for i in range(min(window_width, len(arr))):
        W_sum_prev += weights[i]
        arr_times_W_sum_prev += weights[i] * arr[i]
        mu_next = arr_times_W_sum_prev / W_sum_prev
        if i > 0:
            ss_prev += weights[i] * (arr[i] - mu_prev) * (arr[i] - mu_next)
        mu_prev = mu_next
        out[i] = np.sqrt(ss_prev / W_sum_prev)  # Cumulative standard deviation

    # Continue with the standard weighted moving standard deviation approach
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
        out[i] = np.sqrt(ss_prev / W_sum_next)
        arr_times_W_sum_prev = arr_times_W_sum_next
        W_sum_prev = W_sum_next
        mu_prev = mu_next

    return out
