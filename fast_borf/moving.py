import numpy as np
import numba as nb
from fast_borf.constants import FASTMATH


@nb.njit(fastmath=FASTMATH, cache=True)
def move_diff(a, window_width):
    out = np.full_like(a, np.nan, dtype=np.float64)
    for i in range(len(a) - window_width + 1):
        out[i + window_width - 1] = a[i + window_width - 1] - a[i]
    return out


@nb.njit(fastmath=FASTMATH, cache=True)
def move_mean(a, window_width):
    out = np.empty_like(a)
    asum = 0.0
    count = 0

    # Calculating the initial moving window sum
    for i in range(window_width):
        asum += a[i]
        count += 1
        out[i] = asum / count

    # Moving window
    for i in range(window_width, len(a)):
        asum += a[i] - a[i - window_width]
        out[i] = asum / window_width
    return out


@nb.njit(fastmath=FASTMATH, cache=True)
def move_sum(a, window_width):
    out = np.empty_like(a)
    asum = 0.0

    # Calculating the initial moving window sum
    for i in range(window_width):
        asum += a[i]
        out[i] = asum

    # Moving window
    for i in range(window_width, len(a)):
        asum += a[i] - a[i - window_width]
        out[i] = asum

    return out


@nb.njit(fastmath=FASTMATH, cache=True)
def move_std(a, window_width, ddof=0):
    out = np.empty_like(a, dtype=np.float64)

    # Initial mean and variance calculation (Welford's method)
    mean = 0.0
    M2 = 0.0
    for i in range(window_width):
        delta = a[i] - mean
        mean += delta / (i + 1)
        delta2 = a[i] - mean
        M2 += delta * delta2

    # Adjusting for degrees of freedom
    if window_width - ddof > 0:
        variance = M2 / (window_width - ddof)
    else:
        variance = 0  # Avoid division by zero

    out[window_width - 1] = np.sqrt(max(variance, 0))

    # Moving window
    for i in range(window_width, len(a)):
        x0 = a[i - window_width]
        xn = a[i]

        new_avg = mean + (xn - x0) / window_width

        # Update the variance using the new degrees of freedom
        if window_width - ddof > 0:
            new_var = variance + (xn - new_avg + x0 - mean) * (xn - x0) / (
                window_width - ddof
            )
        else:
            new_var = 0  # Avoid division by zero

        out[i] = np.sqrt(max(new_var, 0))  # TODO: investigate negative variance (this was a bug that I found)

        mean = new_avg
        variance = new_var

    out[: window_width - 1] = np.nan

    return out


@nb.njit(fastmath=FASTMATH, cache=True)
def move_var(a, window_width, ddof=0):
    out = np.empty_like(a, dtype=np.float64)

    # Initial mean and variance calculation (Welford's method)
    mean = 0.0
    M2 = 0.0
    for i in range(window_width):
        delta = a[i] - mean
        mean += delta / (i + 1)
        delta2 = a[i] - mean
        M2 += delta * delta2

    # Adjusting for degrees of freedom
    if window_width - ddof > 0:
        variance = M2 / (window_width - ddof)
    else:
        variance = 0  # Avoid division by zero

    out[window_width - 1] = max(variance, 0)

    # Moving window
    for i in range(window_width, len(a)):
        x0 = a[i - window_width]
        xn = a[i]

        new_avg = mean + (xn - x0) / window_width

        # Update the variance using the new degrees of freedom
        if window_width - ddof > 0:
            new_var = variance + (xn - new_avg + x0 - mean) * (xn - x0) / (
                window_width - ddof
            )
        else:
            new_var = 0  # Avoid division by zero

        out[i] = max(new_var, 0)

        mean = new_avg
        variance = new_var

    out[: window_width - 1] = np.nan

    return out


@nb.njit(fastmath=FASTMATH, cache=True)
def move_cov(x, y, window_width):
    xy = x * y
    move_sum_xy = move_sum(xy, window_width)
    move_sum_x = move_sum(x, window_width)
    move_sum_y = move_sum(y, window_width)
    covariance = (move_sum_xy - (move_sum_x * move_sum_y / window_width)) / window_width
    return covariance




@nb.njit(fastmath=FASTMATH, cache=True)
def move_slope(x, y, window_width):
    move_cov_xy = move_cov(x, y, window_width)
    move_var_x = move_var(x, window_width)
    slope = move_cov_xy / move_var_x
    return slope


@nb.njit(fastmath=FASTMATH, cache=True)
def KahanSum(input_array):
    sum = 0.0
    c = 0.0  # A compensation for lost low-order bits
    for i in range(len(input_array)):
        y = input_array[i] - c
        t = sum + y
        c = (t - sum) - y
        sum = t
    return sum


@nb.njit(fastmath=FASTMATH, cache=True)
def move_std_kahan(a, window_width, ddof=0):
    out = np.empty_like(a, dtype=np.float64)

    # Use Kahan Summation for initial mean calculation
    initial_sum = KahanSum(a[:window_width])
    mean = initial_sum / window_width
    M2 = 0.0
    for i in range(window_width):
        delta = a[i] - mean
        delta2 = a[i] - mean
        M2 += delta * delta2

    # Adjusting for degrees of freedom
    if window_width - ddof > 0:
        variance = M2 / (window_width - ddof)
    else:
        variance = 0  # Avoid division by zero

    out[window_width - 1] = np.sqrt(variance)

    # Moving window
    for i in range(window_width, len(a)):
        x0 = a[i - window_width]
        xn = a[i]

        # Update the mean using a simplified Kahan Summation step
        mean_delta = (xn - x0) / window_width
        new_avg = mean + mean_delta

        if window_width - ddof > 0:
            new_var = variance + (xn - new_avg + x0 - mean) * (xn - x0) / (window_width - ddof)
        else:
            new_var = 0  # Avoid division by zero

        out[i] = np.sqrt(new_var)

        mean = new_avg
        variance = new_var

    out[:window_width - 1] = np.nan

    return out


def move_sum_kahan(a, window_width):
    out = np.empty_like(a, dtype=np.float64)
    n = len(a)

    # Calculate initial sum for the first window using Kahan Summation
    initial_sum = KahanSum(a[:window_width])
    out[window_width - 1] = initial_sum

    # For subsequent windows, update the sum by subtracting the element exiting the window
    # and adding the new element entering the window, applying a simplified Kahan correction.
    c = 0.0  # Reset the compensation for lost low-order bits
    for i in range(window_width, n):
        x0 = a[i - window_width]
        xn = a[i]

        # Update the sum using a simplified Kahan Summation step
        y = xn - x0 - c
        t = out[i - 1] + y
        c = (t - out[i - 1]) - y
        out[i] = t

    # Set the output for indices before the first complete window to NaN
    out[:window_width - 1] = np.nan

    return out


# move_std = move_std_kahan
# move_sum = move_sum_kahan


if __name__ == "__main__":
    arr = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 3.0, 5.0, 8.0, 10.0, 13.0, 17.0, 0.0, 13.0, 0.0, 22.0, 3.0, 5.0, 0.0, 4.0, 0.0, 9.0, 5.0, 2.0])
    window_width = 32
    std_ = move_std(arr, window_width)

    # arr = np.random.randn(50)#, dtype=np.float64)
    # window_width = 2
    # std_1 = move_std(arr, window_width)
    # std_2 = move_std_kahan(arr, window_width)
    # print(np.array_equal(std_1, std_2, equal_nan=True))
    # print(np.allclose(std_1, std_2, equal_nan=True))
    #
    # sum_1 = move_sum(arr, window_width)
    # sum_2 = move_sum_kahan(arr, window_width)
    # print(np.array_equal(sum_1[1:], sum_2[1:], equal_nan=True))
    # print(np.allclose(sum_1[1:], sum_2[1:], equal_nan=True))



    # window_width = 8
    # print(move_mean(arr, window_width)[window_width - 1 :])
    # window_width = 16
    # print(move_mean(arr, window_width)[window_width - 1 :])

    # import bottleneck as bn
    #
    # arr = np.arange(10, dtype=np.float64)
    # x = np.arange(10)
    # y = np.random.rand(10)
    # arr = np.random.rand(5000)
    # # print(arr)
    # # print(move_mean(arr, 3))
    # # print(move_sum(arr, 3))
    # # print(moving_std(arr, 3))
    # # print(bn.move_std(arr, 3, ddof=1))
    # m1 = move_std(arr, 1000, ddof=0)
    # m2 = bn.move_std(arr, 1000, ddof=0)
    # print(np.allclose(m1, m2, equal_nan=True))
    # # print(m1 == m2)
