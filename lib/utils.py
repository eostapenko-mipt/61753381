import copy
import marshal

from tqdm.notebook import trange, tqdm
from IPython.display import clear_output

# enumerate, который производит итерацию начиная с последнего элемента в списке
def reversed_enumerate(arr: list):
    for i in range(len(arr) - 1, -1, -1):
        yield i, arr[i]


# deep_round
# Округление всей матрицы
def deep_round(val, prec=0):
    if not hasattr(val, '__iter__'):
        return round(val, prec)
    for i, _ in enumerate(val):
        val[i] = deep_round(val[i], prec)
    return val


def marshal_deep_copy(obj):
    return marshal.loads(marshal.dumps(obj))


def _deep_copy(obj):
    return copy.deepcopy(obj)
    # return marshal_deep_copy(obj)

def deep_copy(obj):
    return _deep_copy(obj)

# nget
# Вернуть значение сложного объекта по указанному пути
def nget(obj, path, defval=None):
    for k in path:
        if obj is None:
            return defval
        elif hasattr(obj, 'get'):
            if k not in obj:
                return defval
            obj = obj.get(k)
        elif hasattr(obj, '__iter__'):
            if type(k) != int:
                return defval
            elif len(obj) - 1 < k:
                return defval
            else:
                obj = obj[k]
        else:
            return defval
    return obj


# Записать значение сложного объекта по указанному пути
def nset(obj, path, val):
    _obj = nget(obj, path[0:-1])
    if _obj:
        _obj[path[-1]] = val
    return val


# find_rel_val
def find_rel_val(ndarr, glob_item, cmp_func):
    glob_index = tuple()
    for index, item in enumerate(ndarr):
        if hasattr(item, '__iter__'):
            _index, _item = find_rel_val(item, glob_item, cmp_func)
            if cmp_func(_item, glob_item):
                glob_item  = _item
                glob_index = (index, *_index)
            continue
        if cmp_func(item, glob_item):
            glob_item = item
            glob_index = tuple([index])
    return (glob_index, glob_item)


def _min_cmp_func(v, g):
    return v < g


def _max_cmp_func(v, g):
    return v > g


def findmin(ndarr, min_v=float("inf")):
    return find_rel_val(ndarr, min_v, _min_cmp_func)


def findmax(ndarr, max_v=-float("inf")):
    return find_rel_val(ndarr, max_v, _max_cmp_func)


def argmin(arr, min_v=float("inf")):
    i, v = findmin(arr)
    if not len(i):
         return None
    if len(i) == 1:
        return i[0]
    return i


def argmax(arr, min_v=float("inf")):
    i, v = findmax(arr)
    if not len(i):
         return None
    if len(i) == 1:
        return i[0]
    return i


# Конвертировать 2d представление задачи в 3d
def _mat3d(mat2d, shape):
    _res3d = []
    _mat2d = []
    shape = shape.copy()
    p = shape.pop(0)
    for row in mat2d:
        _mat2d.append(row)
        if len(_mat2d) == p:
            _res3d.append(_mat2d)
            if shape:
                p = shape.pop(0)
            _mat2d = []
    return _res3d


# Конвертировать 3d представление задачи в 2d
def _mat2d(mat3d):
    _res2d = []
    for mat in mat3d:
        _res2d.extend(mat)
    return _res2d


def mat3d(mat2d, shape):
    return _mat3d(mat2d, shape)


def mat2d(mat3d):
    return _mat2d(mat3d)


def print_osp_matrix(w, jsep=None):
    _, m = findmax(w)
    pad = len(str(m)) + 1
    pad0 = len(str(m))
    # print("max:", m, l)
    if type(w[0][0]) == int:
        w = [w]
    for mat in w:
        for row in mat:
            rowstr = ""
            for i, col in enumerate(row):
                rowstr += str(col).rjust(pad0 if not i else pad, " ") + ","
            rowstr = "[" + rowstr.strip(",") + "]"
            print(rowstr)
        if jsep:
            print(jsep)


# Рассчитать и вернуть значение целевой функции из решения
# Матричное решение
def calc_obj_from_mat(w, x):
    # w = матрица взвешенного времени
    # x = матрица решения
    # Сумма по всем элементам матрицы произведения Адамара (w * x)
    if not hasattr(w, '__iter__'):
        return w * x
    _sum = 0
    for i, _ in enumerate(w):
        _sum += calc_obj_from_mat(w[i], x[i])
    return _sum


# Рассчитать и вернуть значение целевой функции из решения
# Векторное решение. К примеру эвристики
def calc_obj_from_vec(w, v):
    _sum = 0
    ktable = [0] * len(w)
    for t, j in enumerate(v):
        if j < 0: continue
        j = j - 1
        k = ktable[j]
        _sum += w[j][k][t]
        ktable[j] += 1
    return _sum