from munkres import Munkres, print_matrix
from scipy.optimize import linear_sum_assignment

from lib.utils import (
    mat2d,
    mat3d,
    deep_round,
    _deep_copy,
    calc_obj_from_mat,
    calc_obj_from_vec,
)

# Решить задачу о назначениях
# Алгоритм из пакета scipy
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
# https://ieeexplore.ieee.org/document/7738348
# Решение будет оптимальным, но может отличаться от венгерского
def solve_lap_scipy(w):
    # w: list[T,T] = матрица взвешенного времени

    sol_index_x, sol_index_y = linear_sum_assignment(w)
    n, m = len(sol_index_x), len(sol_index_y)
    x = [[0] * m for i in range(n)]
    for k, _ in enumerate(sol_index_x):
        i = sol_index_x[k]
        j = sol_index_y[k]
        x[i][j] = 1
    return x


__global_munk = Munkres()

# Решить задачу о назначениях
# Венгерский алгоритм
# Также известный как Kuhn–Munkres algorithm или Munkres assignment algorithm
# https://en.wikipedia.org/wiki/Hungarian_algorithm
def solve_lap_munkres(_w):
    # _w: list[T,T] = матрица взвешенного времени

    w = _w.copy()
    indexes = __global_munk.compute(w)
    n = len(w)
    m = len(w[0])
    x = [[0] * m for i in range(n)]
    for i,j in indexes:
        x[i][j] = 1
    return x


# Рассчитать верхний допуск ЗН
def get_lap_tolerance(w, x, i, j, solver=solve_lap_munkres):
    obj = calc_obj_from_mat(w, x)
    _inf = w[1][0]
    _w = _deep_copy(w)
    _w = exclude_weight_2d(_w, i, j, _inf)
    _x = solver(_w)
    _obj = calc_obj_from_mat(_w, _x)
    return _obj - obj, _w, _x, _obj


# Рассчитать верхние допуски ЗН
# Вернуть список от большего к меньшему
def get_lap_tolerances(w, x, inf_def=0, solver=solve_lap_munkres):
    tolerances = []
    obj = calc_obj_from_mat(w, x)
    _inf = w[1][0]
    for i, _ in enumerate(x):
        for j, _ in enumerate(x[i]):
            if x[i][j] > 0:
                _w = _deep_copy(w)
                _w = exclude_weight_2d(_w, i, j, _inf)
                _x = solver(_w)
                _obj = calc_obj_from_mat(_w, _x)
                if _obj >= _inf:
                    if inf_def:
                        tolerances.append((
                            (i, j),
                            inf_def,
                        ))
                    continue
                tolerances.append((
                    (i, j),
                    (_obj - obj)
                ))
    return sorted(tolerances, key=_func1172_toler_sort, reverse=True)

def _func1172_toler_sort(a):
    return a[1]


# Форсировать исключение коэф. из решения:
# задать достаточно большое число указанному коэф.
def exclude_weight_2d(w, i, j, _inf):
    w[i][j] = _inf
    return w


# Форсировать включение коэф. в решение:
# задать достаточно большое число всем коэф. кроме указанного
def include_weight_2d(*args):
    return include_weight_r_2d(*args)


def include_weight_r_2d(w, i, j, _inf):
    v = w[i][j]
    for jj, _ in enumerate(w[i]):
        w[i][jj] = _inf
    w[i][j] = v
    return w


def get_weight_inf_value_2d(w):
    return w[1][0]