from typing import List

import numpy as np


def create_dropout_weights(weight_matrix_list: List[np.array], dropout: float):
    ret = []
    for w in weight_matrix_list:
        nrow = w.shape[0]
        ncol = w.shape[1]
        sz = nrow * ncol
        n = int(np.rint(dropout * sz))

        print(f'sz = {sz}')
        print(f'n = {n}')
        B = np.ones((sz, 1), dtype=np.float64)
        idx = np.random.choice(sz, size=n, replace=False)
        B[idx] = 0.0

        B = B.reshape((nrow, ncol))
        ret.append(w * B)
    return ret


if __name__ == '__main__':
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = np.array([[1], [2], [3]])
    C = np.array([[1, 2, 3]])
    lst = [A, B, C]

    ret = create_dropout_weights(lst, 0.5)
    for w in ret:
        print(w)
