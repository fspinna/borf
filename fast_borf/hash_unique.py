import numba
import numpy as np


@numba.njit(cache=True)
def length(l):
    l = int(np.ceil(np.log2(l)))
    # 4*len(ar) > l > 2*len(ar)
    l = 2 << l
    return l


@numba.njit(cache=True)
def hash_function(v):

    byte_mask = np.uint64(255)
    bs = np.uint64(v)
    x1 = (bs) & byte_mask
    x2 = (bs >> 8) & byte_mask
    x3 = (bs >> 16) & byte_mask
    x4 = (bs >> 24) & byte_mask
    x5 = (bs >> 32) & byte_mask
    x6 = (bs >> 40) & byte_mask
    x7 = (bs >> 48) & byte_mask
    x8 = (bs >> 56) & byte_mask

    FNV_primer = np.uint64(1099511628211)
    FNV_bias = np.uint64(14695981039346656037)
    h = FNV_bias
    h = h * FNV_primer
    h = h ^ x1
    h = h * FNV_primer
    h = h ^ x2
    h = h * FNV_primer
    h = h ^ x3
    h = h * FNV_primer
    h = h ^ x4
    h = h * FNV_primer
    h = h ^ x5
    h = h * FNV_primer
    h = h ^ x6
    h = h * FNV_primer
    h = h ^ x7
    h = h * FNV_primer
    h = h ^ x8
    return h


@numba.njit(cache=True)
def make_hash_table(ar):
    l = length(len(ar))
    mask = l - 1

    uniques = np.empty(l, dtype=ar.dtype)
    uniques_cnt = np.zeros(l, dtype=np.int_)
    return uniques, uniques_cnt, l, mask

@numba.njit(cache=True)
def set_item(uniques, uniques_cnt, mask, h, v, total, miss_hits, weight):
    index = (h & mask)

    # open address hash
    # great cache performance
    while True:
        if uniques_cnt[index] == 0:
            # insert new
            uniques_cnt[index] += weight
            uniques[index] = v
            total += 1
            break
        elif uniques[index] == v:
            uniques_cnt[index] += weight
            break
        else:
            miss_hits += 1
            index += 1
            index = index & mask
    return total, miss_hits


@numba.njit(cache=True)
def concrete(ar, uniques, uniques_cnt, l, total):
    # flush the results in a concrete array
    uniques_ = np.empty(total, dtype=ar.dtype)
    uniques_cnt_ = np.empty(total, dtype=np.int_)
    t = 0
    for i in range(l):
        if uniques_cnt[i] > 0:
            uniques_[t] = uniques[i]
            uniques_cnt_[t] = uniques_cnt[i]
            t += 1
    return uniques_, uniques_cnt_


@numba.njit(cache=True)
def unique(ar):
    uniques, uniques_cnt, l, mask = make_hash_table(ar)
    total = 0
    miss_hits = 0
    for v in ar:
        h = hash_function(v)
        total, miss_hits = set_item(uniques, uniques_cnt, mask, h, v, total, miss_hits, 1)
    uniques_, uniques_cnt_ = concrete(ar, uniques, uniques_cnt, l, total)
    return uniques_, uniques_cnt_
