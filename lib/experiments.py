import os
import csv
import math
import copy
import json
import time
import random
import asyncio
import hashlib
import networkx as nx
import itertools
import subprocess
import aiofiles
import aiofiles.os
import matplotlib.pyplot as plt
import datetime
import inspect
import traceback
import marshal

from munkres import Munkres, print_matrix
from functools import cmp_to_key
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

import lib.utils
import lib.config
import lib.lap

from lib.utils import (
    calc_obj_from_mat,
    calc_obj_from_vec,
    reversed_enumerate,
    print_osp_matrix,
    _deep_copy,
    _mat2d,
    _mat3d,
)

from lib.lap import (
    get_weight_inf_value_2d,
    include_weight_2d,
    exclude_weight_2d,
    get_lap_tolerance,
)

from lib.lp import (
    solve_lp,
)

INF = float("inf")

# Записать тест и его результаты на диск
async def write_instance_into_disk(
    instance,
    data_path=None
):
    problem_instances_path = data_path or lib.config.global_config["default_problem_instances_path"]
    
    params  = instance["params"]
    n       = len(params["p"])
    p       = params["p"][0]
    npkey   = f"n{n}p{p}"

    _dirpath  = problem_instances_path + "/json/" + npkey
    _filepath = _dirpath + "/" + instance["id"] + ".json"

    os.makedirs(_dirpath, exist_ok=True)

    async with aiofiles.open(_filepath, "w") as fd:
        await fd.write(json.dumps(instance))


# BaseProblemInstanceLoader
# Загрузчик датасета с диска
class BaseProblemInstanceLoader:
    def __init__(self, path):
        self._path = path
        self._items = None
        self._curr = None
        self._dirs = os.listdir(self._path)
        self._dirs = list(
            filter(
                lambda x: x[0] != "_",
                self._dirs,
            )
        )

    def __iter__(self):
        return self
        
    def __next__(self):
        if (not self._dirs) and (not self._items):
            raise StopIteration

        if not self._items:
            _dirname    = self._dirs.pop(0)
            self._curr  = os.path.join(self._path, _dirname)
            self._items = os.listdir(self._curr)

        _filename = self._items.pop(0)
        _filepath = os.path.join(self._curr, _filename) 

        fd = open(_filepath, "r", encoding="utf-8")
        data = json.load(fd)
        fd.close()

        return data

class PILStd(BaseProblemInstanceLoader):
    def __next__(self):
        instance  = super().__next__()
        # instance["results"]["all_intime"] = _instance_has_all_intime(instance)
        # instance["results"]["intersec"]   = _instance_has_mutual_job_intersection(instance)

        return instance

# Вариант без решений -- меньше памяти
class PILVerbose0(PILStd):
    def __next__(self):
        instance = super().__next__()
        for test_name in instance["results"]:
            if "x" in instance["results"][test_name]:
                instance["results"][test_name]["x"] = None
        return instance


def _mat2d_lap_sol_into_vec(x_mat2d):
    vec = [-1] * len(x_mat2d)
    for i, _ in enumerate(x_mat2d):
        for j, _ in enumerate(x_mat2d[i]):
            if x_mat2d[i][j]:
                vec[i] = j
                break
    return vec


def mat2d_lap_sol_into_vec(x_mat2d):
    return _mat2d_lap_sol_into_vec(x_mat2d)


def _vec_lap_sol_into_mat2d(vec):
    n = len(vec)
    mat2d = [[0] * n for _ in range(n)]
    for i, j in enumerate(vec):
        mat2d[i][j] = 1
    return mat2d


def vec_lap_sol_into_mat2d(vec):
    return _vec_lap_sol_into_mat2d(vec)


def mat3d_lp_sol_into_vec(x_mat3d):
    res = []
    for j, mat in enumerate(x_mat3d):
        for k, row in enumerate(x_mat3d[j]):
            for t, val in enumerate(x_mat3d[j][k]):
                if x_mat3d[j][k][t]:
                    res.append(
                        (
                            (j, k, t),
                            x_mat3d[j][k][t]
                        )
                    )
    return res

def vec_lp_sol_into_mat3d(vec, params):
    mat3d = []
    h = 0
    for pj in params["p"]:
        h = h + pj
    for pj in params["p"]:
        mat3d.append([[0] * h for _ in range(pj)])
    for item in vec:
        ((j, k, t), val) = item
        mat3d[j][k][t] = val
    return mat3d


# Собрать уникальный id для экземпляра теста
def create_problem_instance_id(instance):
    _str = ""
    params = instance["params"]
    _keys = sorted(params.keys(), reverse=False)
    for k in _keys:
        _str += str(k) + "=" + str(params[k]) + ";"
    return hashlib.md5(_str.encode()).hexdigest()


# Иной вариант
def generate_test_case_params_twct(n, p):
    # n: int = количество работ
    # p: int = количество операций в работах

    params = {}
    params["p"]  = []
    params["w"]  = []
    params["d1"] = [0] * n
    params["d2"] = [0] * n
    params["s1"] = [0] * n
    params["s2"] = [n*p+1] * n

    params["r"]  = list(range(1, n + 1))
    random.shuffle(params["r"])
    
    for j in range(n):
        r = params["r"][j]
        w = random.randint(1, 30)
        params["p"].append(p)
        params["w"].append(w)

    return params


def generate_test_case_params_twscp(n, p):
    # n: int = количество работ
    # p: int = количество операций в работах

    params = {}
    params["p"]  = []
    params["w"]  = []
    params["d1"] = []
    params["d2"] = []
    params["s1"] = []
    params["s2"] = []

    params["r"]  = list(range(1, n + 1))
    random.shuffle(params["r"])
    
    for j in range(n):
        # r = random.randint(1, n * p - p);
        r = params["r"][j]
        w = random.randint(1, 30)
        d1 = random.randint(r + p, n * p)
        d2 = random.randint(d1, n * p)
        s1 = random.randint(r, d1)
        s2 = random.randint(s1, d1)
        params["p"].append(p)
        # params["r"].append(r)
        params["w"].append(w)
        params["d1"].append(d1)
        params["d2"].append(d2)
        params["s1"].append(s1)
        params["s2"].append(s2)

    return params


# Вернуть тестовые параметры. Пример из диплома
def get_test_case_1_params():
    return {
        "p" : [3, 2, 5],
        "w" : [1, 3, 5],
        "r" : [1, 2, 3],
        "s1": [3, 5, 5],
        "s2": [3, 6, 6],
        "d1": [4, 7, 9],
        "d2": [7, 7, 9],
    }


# Вернуть параметры example 2. Из "HOCO Scheduling Final 08.07.2024.pdf"
def get_test_case_2_params():
    return {
        "p":  [2,  2,  2,  2],
        "w" : [4,  9, 12,  9],
        "r" : [1,  4,  3,  2],
        "s1": [0,  0,  0,  0],
        "s2": [8,  8,  8,  8],
        "d1": [0,  0,  0,  0],
        "d2": [0,  0,  0,  0],
    }


# Матричное представление задачи
# Рассчитать и вернуть матрицу взвешенного времени (матрицу стоимостей)
# 
def create_cost_matrix_twscp(p, w, r, s1, s2, d1, d2):
    # p:  list = PROC_TIMES     = Количество задач в работах
    # w:  list = WEIGHTS        = Веса
    # r:  list = RELEASE_DATES  = Время старта работы
    # s1: list = EARLY_S_DATES  = Время опережения запуска работы
    # s2: list = DUE_S_DATES    = Время запаздывания запуска работы
    # d1: list = EARLY_C_DATES  = Время опережения завершения работы
    # d2: list = DUE_C_DATES    = Время запаздывания завершения работы

    n    = len(w) # количество работ
    T    = sum(p) # временной горизонт
    _inf = max((T*max(w))**2, 1000) # достаточно большое число (выполняет роль бесконечности)

    # результат, матрица взвешенного времени
    # инициализировать с нулями
    cost = []

    for j in range(n):
        job = []
        
        for k in range(p[j]):
            oper = []
            _k = k + 1  # индекс работы начиная от 1 (в соотв. с описанием в задаче ЛП)
            
            for t in range(T):
                _t = t + 1  # время начиная от 1 (в соотв. с описанием в задаче ЛП)
                
                # w_jpt = последняя операция в работе
                if _k == p[j]:
                    if r[j] + p[j] - 1 <= _t:
                        w_jkt = w[j] * (  max(0, _t - d2[j]) + max(0, d1[j] - _t)  )
                    else:
                        w_jkt = _inf
                        
                # w_j1t = первая операция в работе
                elif _k == 1:
                    if r[j] <= _t and _t <= T - p[j] + 1:
                        w_jkt = w[j] * (  max(0, _t - s2[j]) + max(0, s1[j] - _t)  )
                    else:
                        w_jkt = _inf
                        
                # w_jkt = все остальные операции
                else:
                    if r[j] + _k - 1 <= _t and _t <= T - p[j] + _k:
                        w_jkt = 0
                    else:
                        w_jkt = _inf
                        
                oper.append(w_jkt)
            # end for t
            job.append(oper)
        # end for k
        cost.append(job)
    # end for j

    return cost


def get_schedule_vec(x):
    # x: list[j,k,t]
    vec = [0] * len(x[0][0])
    for j, _ in enumerate(x):
        for k, _ in enumerate(x[j]):
            for t, _ in enumerate(x[j][k]):
                if x[j][k][t]:
                    vec[t] = j + 1
    return vec


# Нарисовать ДПОР для АВиГ ЗН
def draw_sst(tree):
    g = nx.Graph()

    edge_labels = {}
    node_color = []

    stack = [tree]
    parent = None

    while stack:
        node = stack.pop()
        node_name = str(node["i"]) + "_" +  str(node["obj"])
        g.add_node(node_name)
        color = "black"
        if node.get("feasible"):
            color = "tab:blue"
        if node.get("lb_bu_geq_ub"):
            color = "darkviolet"
        if node.get("lb_ge_ub"):
            color = "deeppink"
        if node.get("lb_geq_inf"):
            color = "darkred"
        if node.get("out_of_options"):
            color = "gray"
        node_color.append(color)
        
        if not node["nodes"]:
            continue

        left  = node["nodes"][0]
        right = node["nodes"][1]

        # left
        left_name = str(left["i"]) + "_" +  str(left["obj"])
        stack.insert(0, left)
        g.add_edge(node_name, left_name)
        edge_labels[(node_name, left_name)] = "--" + str(left["arc"])

        # right
        right_name = str(right["i"]) + "_" +  str(right["obj"])
        stack.insert(0, right)
        g.add_edge(node_name, right_name)
        edge_labels[(node_name, right_name)] = "+" + str(right["arc"])

    pos = nx.nx_pydot.pydot_layout(g, prog="dot")

    plt.figure(
        num=None,
        figsize=(12, 20),
        # dpi=120
    )

    nx.draw(
        g,
        font_color="white",
        node_color=node_color,
        with_labels=True,
        pos=pos,
        node_size=800,
        font_size=8,
    )
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_edge_labels(
        g,
        pos,
        edge_labels=edge_labels,
        font_color='red',
        font_size=8,
    )


def solve_osp_lap_bnb_2d_v3_3(
    w,
    ctx=None,
    _parent_node=None,
    _arc=None,
    _is_exclusion=None,
    _ancestors=None,
    _cache_x_lap=None
):
    if "t0" not in ctx:
        ctx["t0"] = time.process_time()

    if "t_stop_limit" in ctx:
        if (time.process_time() - ctx["t0"]) >= ctx.get("t_stop_limit", INF):
            raise BaseException("t_stop_limit")

    if _ancestors is None:
        _ancestors = dict()

    _solver = ctx["solver"]
    _print_debug = ctx.get("print_debug", 0)
    _inf = get_weight_inf_value_2d(w)

    _is_inclusion = not _is_exclusion

    if _is_exclusion and _cache_x_lap and ctx.get("reuse_parent_x_for_exclusion", 1):
        x_lap = _cache_x_lap
    elif _is_inclusion and _cache_x_lap and ctx.get("reuse_parent_x_for_inclusion", 1):
        x_lap = _cache_x_lap
    else:
        x_lap = _solver(w)
        ctx["n_lap_solved"] = ctx.get("n_lap_solved", 0) + 1

    ctx["n_tree_nodes"] = ctx.get("n_tree_nodes", 0) + 1

    # if (ctx["n_tree_nodes"] % 1000 == 0):
    #     print(ctx["n_tree_nodes"])

    if ctx["n_tree_nodes"] >= ctx.get("n_tree_nodes_stop_limit", INF):
        raise BaseException("n_tree_nodes_stop_limit")

    obj = calc_obj_from_mat(w, x_lap)
    lb = obj

    node = None

    if ctx.get("build_tree", 0):
        node = {
            "i"    : ctx["n_tree_nodes"],
            "arc"  : _arc,
            "obj"  : obj,
            "nodes": [],
            "is_exclusion": _is_exclusion,
        }

    if _parent_node and node:
        _parent_node["nodes"].append(node)

    if _print_debug:
        print()
        print(f"arc:{str(_arc)}; i:{ctx['n_tree_nodes']}; obj:{obj}; ub:{ctx['ub'][1]}")
        print_osp_matrix(x_lap)

    if "solution_search_tree" not in ctx:
        ctx["solution_search_tree"] = node

    if lb > ctx["ub"][1]:
        ctx["n_cut_obj_ub"] = ctx.get("n_cut_obj_ub", 0) + 1
        ctx["n_cut_lb_ub"] = ctx.get("n_cut_lb_ub", 0) + 1
        _note = f"obj > ub: {obj} > {ctx['ub'][1]}"
        if node:
            node["note"] = _note
            node["lb_ge_ub"] = 1
        if _print_debug:
            print(_note)
        return ctx["ub"][0]

    if lb >= _inf:
        ctx["n_cut_inf"] = ctx.get("n_cut_inf", 0) + 1
        _note = f"obj >= _inf: {obj} > {_inf}"
        if node:
            node["note"] = _note
            node["lb_geq_inf"] = 1
        if _print_debug:
            print(_note)
        return ctx["ub"][0]

    infeasible_arcs = get_schedule_infeasible_arcs(x_lap, ctx["params"])

    if infeasible_arcs is None: # isValidSchedule
        ctx["n_cut_feasible"] = ctx.get("n_cut_feasible", 0) + 1
        ctx["ub"] = (x_lap, obj)
        if node:
            node["note"] = f"feasible: {obj}"
            node["feasible"] = 1
        if _print_debug:
            print('feasible:', ctx["ub"][1]);
        return x_lap

    branch_arc = None

    if "lowest_cost_arc" == ctx.get("arc_branch_rule"):
        for arc in infeasible_arcs:
            if (arc in _ancestors):
                continue
            if branch_arc is None:
                branch_arc = arc
                continue
            ci, cj = arc
            mi, mj = branch_arc
            if (w[ci][cj] < w[mi][mj]):
                branch_arc = arc
                
    elif "highest_cost_arc" == ctx.get("arc_branch_rule"):
        for arc in infeasible_arcs:
            if (arc in _ancestors):
                continue
            if branch_arc is None:
                branch_arc = arc
                continue
            ci, cj = arc
            mi, mj = branch_arc
            if (w[ci][cj] > w[mi][mj]):
                branch_arc = arc
                
    elif "first_available_arc" == ctx.get("arc_branch_rule"):
        for arc in infeasible_arcs:
            if (arc in _ancestors):
                continue
            branch_arc = arc
            break
    else:
        raise BaseException("unknown branch_rule " + ctx.get("arc_branch_rule"))
    
    # out of options
    if branch_arc is None:
        ctx["n_cut_no_opt"] = ctx.get("n_cut_no_opt", 0) + 1
        _note = (f""
            + f"branch_arc is None (out of options):"
            + f" {str(_arc)};"
            + f" i:{ctx['n_tree_nodes']};"
            + f" infeasible_arcs: {str(infeasible_arcs)};"
        )
        if node:
            node["note"] = _note
            node["out_of_options"] = 1
        if _print_debug:
            print(_note)
        return ctx["ub"][0]

    w1 = x_lap_1 = None

    if ctx.get("tolerance_branch_rule"):
        if "lowest_upper_tolerance" == ctx.get("tolerance_branch_rule"):
            branch_arc = None
            branch_tol = None
            for arc in infeasible_arcs:
                if (arc in _ancestors):
                    continue
                _curr_tol = get_lap_tolerance(w, x_lap, *arc, ctx["params"], solver=_solver)
                ctx["n_created_w"] = ctx.get("n_created_w", 0) + 1
                ctx["n_lap_solved"] = ctx.get("n_lap_solved", 0) + 1
                if branch_arc is None:
                    branch_arc = arc
                    branch_tol = _curr_tol
                    continue
                if _curr_tol[0] < branch_tol[0]:
                    branch_arc = arc
                    branch_tol = _curr_tol
            tol, w1, x_lap_1, _ = branch_tol

        elif "arc_upper_tolerance" == ctx.get("tolerance_branch_rule"):
            tol, w1, x_lap_1, _ = get_lap_tolerance(w, x_lap, *branch_arc, ctx["params"], solver=_solver)
            ctx["n_created_w"] = ctx.get("n_created_w", 0) + 1
            ctx["n_lap_solved"] = ctx.get("n_lap_solved", 0) + 1

        else:
            raise BaseException("unknown tolerance_rule" + ctx.get("tolerance_branch_rule"))

        lb = obj + tol

        if not (lb < ctx["ub"][1]):
            ctx["n_cut_lb_ub"] = ctx.get("n_cut_lb_ub", 0) + 1
            _note = f"tolerance{str(branch_arc)}: {obj} + {tol} < {ctx['ub'][1]}"
            if node:
                node["lb_bu_geq_ub"] = 1
                node["note"] = _note
            if _print_debug:
                print(_note)
            return ctx["ub"][0]

    #  and (_is_exclusion or ctx["n_tree_nodes"] == 1)
    if ctx.get("use_swap_patch", 0):
        _x_lap = _deep_copy(x_lap)
        _x_lap = _dev_patch_swap_infeas(_x_lap, ctx["params"])
        _infeasible_arcs = get_schedule_infeasible_arcs(_x_lap, ctx["params"])
        if _infeasible_arcs is None: # isValidSchedule
            _obj = calc_obj_from_mat(w, _x_lap)
            if _obj < ctx["ub"][1]:
                ctx["ub"] = (_x_lap, _obj)
                # x_lap = _x_lap
            if not (lb < ctx["ub"][1]):
                ctx["n_cut_lb_ub"] = ctx.get("n_cut_lb_ub", 0) + 1
                return ctx["ub"][0]

    if not w1:
        w1 = exclude_weight_2d(_deep_copy(w), *branch_arc, _inf)
        ctx["n_created_w"] = ctx.get("n_created_w", 0) + 1

    w2 = include_weight_2d(_deep_copy(w), *branch_arc, _inf)
    ctx["n_created_w"] = ctx.get("n_created_w", 0) + 1
    
    _ancestors = _ancestors.copy()
    _ancestors[branch_arc] = 1

    ctx["n_branching"] = ctx.get("n_branching", 0) + 1

    x_osp_1 = solve_osp_lap_bnb_2d_v3_3(
        w1,
        ctx=ctx,
        _arc=branch_arc,
        _is_exclusion=1,
        _parent_node=node,
        _cache_x_lap=x_lap_1,
        _ancestors=_ancestors
    )

    x_osp_2 = solve_osp_lap_bnb_2d_v3_3(
        w2,
        ctx=ctx,
        _arc=branch_arc,
        _is_exclusion=0, # is_inclusion=1
        _parent_node=node,
        _cache_x_lap=x_lap,
        _ancestors=_ancestors
    )

    if x_osp_1 is None:
        return x_osp_2
    if x_osp_2 is None:
        return x_osp_1

    obj_1 = calc_obj_from_mat(w1, x_osp_1)
    obj_2 = calc_obj_from_mat(w2, x_osp_2)
    
    return x_osp_1 if obj_1 < obj_2 else x_osp_2


# Решить задачу ЛП
async def solve_osp_lap_bnb_2d_v4_0(
    w,
    ctx,
    data_path=None,
):
    should_delete_data_file = False

    # data_input_str = str(len(ctx["params"]["p"])) + "\n";
    data_input_str = "";

    param_keys = ["p", "w", "r", "s1", "s2", "d1", "d2"]

    for k in param_keys:
        data_input_str += " ".join(map(str, ctx["params"][k])) + "\n"
    
    for row in w:
        data_input_str += " ".join(map(str, row)) + "\n"

    data_input_str = data_input_str.strip()

    if not data_path:
        should_delete_data_file = True
        data_path = lib.config.global_config["default_minizinc_data_dir"] + "/" + str(random.random())[3:] + ".txt"

    async with aiofiles.open(data_path, "w") as fd:
        await fd.write(data_input_str)

    cmd = lib.config.global_config["bnb_lap_solver_exec_path"];

    if ctx.get("arc_branch_rule"):
        cmd += " --" + ctx.get("arc_branch_rule")
        
    if ctx.get("tolerance_branch_rule"):
        cmd += " --" + ctx.get("tolerance_branch_rule")
        
    cmd += " " + data_path

    # proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    # (out, err) = proc.communicate()
    out, err = await proc.communicate()

    if err:
        print(err)

    # print(cmd);
    # print(out)
    
    res = json.loads(out)
    
    ctx.update(res)
    d = len(res["x"])
    x_vec = res["x"]
    x_mat = [0] * d

    ctx["exec_time"] = res["exec_time_ms"] / 1000
    
    for i, c in enumerate(x_vec):
        x_mat[i] = [0] * d
        x_mat[i][c] = 1

    if should_delete_data_file:
        await aiofiles.os.remove(data_path)

    return x_mat


def get_wsrpt_upper_bound(w, params):
    # w: list[j, k, t]
    # params: dict
    s = []
    rt = [len(mat) for mat in w]
    _inf = w[0][1][0]
    obj = 0
    for t, _ in enumerate(w[0][0]):
        m = (0, 0, t, -1)
        for j, _ in enumerate(w):
            if not rt[j]:
                continue
            for k, _ in enumerate(w[j]):
                if w[j][k][t] >= _inf:
                    continue
                if k != (params["p"][j] - rt[j]):
                    continue
                q = params["w"][j] / rt[j]
                if q > m[3]:
                    m = (j, k, t, q)
        rt[m[0]] -= 1
        obj += w[m[0]][m[1]][m[2]]
        s.append(m[0] + 1)
    return (s, obj)


# Вернуть время, в которое выполняется каждая работа
def get_job_time_slots(s):
    # s: list = расписание
    job_times = defaultdict(list)
    for t, j in enumerate(s, start=1):
        job_times[j].append(t)
    return job_times


# Проверить прерывает ли работа j2 работу j1
def is_job_intersects(j2, j1, job_times):
    # j1: int = номер прерываемой работы
    # j2: int = номер прерывающей работы
    # job_times: get_job_time_slots()
    if not job_times[j1] or not job_times[j2]:
        return False
    start = job_times[j1][0]
    compl = job_times[j1][-1]
    for t in job_times[j2]:
        if start < t < compl:
            return True
    return False


# Проверить наличие взаимных прерываний
def has_mutual_job_intersection(s, job_times=None):
    # s: list = расписание
    # job_times: get_job_time_slots()
    if not job_times:
        job_times = get_job_time_slots(s)
    for j1, j2 in itertools.combinations(job_times.keys(), 2):
        if (
                is_job_intersects(j2, j1, job_times)
            and is_job_intersects(j1, j2, job_times)
        ):
            return True
    return False


# Проверить наличие взаимных прерываний
def has_job_preemptions(s, job_times=None):
    # s: list = расписание
    # job_times: get_job_time_slots()
    if not job_times:
        job_times = get_job_time_slots(s)
    for j1, j2 in itertools.combinations(job_times.keys(), 2):
        if (
                is_job_intersects(j2, j1, job_times)
            or is_job_intersects(j1, j2, job_times)
        ):
            return True
    return False


# Проверить выполняется ли работа вовремя: вовремя запускается и вовремя завершается
def is_job_processed_intime(j, params, job_times):
    # j: int = номер работы
    # params: object = параметры работ
    # job_times: get_job_time_slots()
    if not job_times:
        return False
    start = job_times[j][0]
    compl = job_times[j][-1]
    h = j - 1
    if (
            (params["s1"][h] <= start <= params["s2"][h])
        and (params["d1"][h] <= compl <= params["d2"][h])
    ):
        return True
    return False


# Проверить выполняемость вовремя работ. Вернуть список [True, False, ...]
def get_processed_intime_jobs(s, params, job_times=None):
    # s: list = расписание
    # params: object = параметры работ
    # job_times: get_job_time_slots()
    if not job_times:
        job_times = get_job_time_slots(s)
    jobs = sorted(job_times.keys(), reverse=False)
    return [
        is_job_processed_intime(j, params, job_times)
        for j in jobs
    ]


# get_positive_x_overlap_ratio
# Вернуть количество переменных которые положительны в обоих решениях
def get_positive_x_overlap_ratio(xi, xf):
    n_overlap = 0
    n_total_i = 0
    # n_total_f = 0
    for j, _ in enumerate(xi):
        for k, _ in enumerate(xi[j]):
            for t, _ in enumerate(xi[j][k]):
                xi_jkt = round(xi[j][k][t], 10)
                xf_jkt = round(xf[j][k][t], 10)
                # if xf_jkt > 0:
                #     n_total_f += 1
                if xi_jkt > 0:
                    n_total_i += 1
                if xi_jkt > 0 and xf_jkt > 0:
                    n_overlap += 1
    return n_overlap / n_total_i


# Вернуть наивную оценку сверху
def get_eager_upper_bound(w):
    _inf = w[0][1][0]
    h = len(w[0][0])
    s = [-1] * h
    ub = 0;
    x = []
    for j, _ in enumerate(w):
        x_job = []
        x.append(x_job)
        for k, _ in enumerate(w[j]):
            x_oper = [0] * h
            x_job.append(x_oper)
            for t, _ in enumerate(w[j][k]):
                if w[j][k][t] < _inf and s[t] == -1:
                    s[t] = j
                    ub += w[j][k][t]
                    x_oper[t] = 1
                    break
    return x, ub


def get_eager_upper_bound_2d(w, params):
    ub = get_eager_upper_bound(_mat3d(w, params["p"]))
    return (_mat2d(ub[0]), ub[1])


def get_schedule_infeasible_arcs(x, params):
    # x: list[j,k,t] = решение
    
    T = len(x[0])
    prev_t = -1
    j = 0
    k = 0
    for a, _ in enumerate(x):
        f = False
        for t in range(prev_t + 1, T):
            if x[a][t]:
                prev_t = t
                f = True
                break
        if not f:
            for t, _ in enumerate(x[a]):
                if x[a][t]:
                    return [(a, t), (a - 1, prev_t)]
        if params["p"][j] - 1 == k:
            prev_t = -1
            j += 1
            k = 0
        else:
            k += 1
    return None


def _dev_patch_swap_infeas(x, params):
    for _ in range(len(x)):
        arcs = get_schedule_infeasible_arcs(x, params)
        if not arcs:
            return x
        (i1, j1), (i2, j2) = arcs
        x[i1][j1] = 0
        x[i2][j2] = 0
        x[i2][j1] = 1
        x[i1][j2] = 1
    return x


# 
# Рассчитать и вернуть эвристику A1
# 
async def solve_a1(w, x, params, opts=None):
    # w: list[j,k,t]
    # x: list[j,k,t] | None
    # params: object = { "w": [...], "r": [...], "p": [...], ... }
    
    # Шаг 1. Решить релаксированную задачу ЛП
    if not x:
        x = await solve_lp(w, "./model_lp.mzn")

    jobs_integer = {}
    jobs_float   = {}

    p             = [len(_x) for _x in x]
    schedule      = [-1] * sum(p)

    opts  = opts or dict()
    _mode = opts.get("mode", "completion_time")
    _prec = opts.get("round", 10)

    # Шаг 2. Определить J_I, J_F
    for j, _ in enumerate(x):
        is_float = False
        max_t = None
        sol_t = []
        pj = len(x[j])
        for k, _ in enumerate(x[j]):
            for t, _ in reversed_enumerate(x[j][k]):
                x_jkt = round(x[j][k][t], _prec)
                if 0 < x_jkt and x_jkt < 1:
                    is_float = True
                if x_jkt:
                    sol_t.append(t)
                if "completion_time" == _mode:
                    if (x_jkt > 0) and (k + 1 == pj) and (not max_t):
                        max_t = t
                elif "starting_time" == _mode:
                    if (x_jkt > 0) and (k == 0) and (not max_t):
                        max_t = t
        if is_float:
            jobs_float[j] = max_t
        else:
            jobs_integer[j] = sol_t

    # Шаг 3. Распланировать J_I как в решении
    for j in jobs_integer:
        for t in jobs_integer[j]:
            schedule[t] = j + 1

    # Шаг 4. Получить "оценку" времени завершения.
    #        Уже предварительно рассчитано на втором шаге.

    # Шаг 5. Распланировать J_F
    def _fsort(a, b):
        j1, j2, t1, t2 = a[0], b[0], a[1], b[1]
        if t1 < t2:
            return -1
        elif t1 == t2:
            if params["w"][j1] > params["w"][j2]:
                return -1
            elif params["w"][j1] == params["w"][j2]:
                return -1 if j1 < j2 else 1
            else:
                return 1
            return -1 if params["w"][j1] < params["w"][j2] else 1
        else:
            return 1

    jobs_float_sorted = sorted(
        jobs_float.items(),
        key=cmp_to_key(_fsort),
        reverse=False
    )

    for j, _ in jobs_float_sorted:
        rj = params["r"][j] - 1
        pj = params["p"][j]
        for t in range(rj, len(schedule)):
            if schedule[t] < 0:
                schedule[t] = j + 1
                pj -= 1
            if not pj:
                break
    
    return schedule


async def solve_a2(w, x, params, opts=None):
    # w: list[j,k,t]
    # x: list[j,k,t] | None
    # params: object = { "w": [...], "r": [...], "p": [...], ... }
    
    # Шаг 1. Решить релаксированную задачу ЛП
    if not x:
        x = await solve_lp(w, "./model_lp.mzn")

    jobs_integer = {}
    jobs_float   = {}
    
    p             = [len(_x) for _x in x]
    schedule      = [-1] * sum(p)
    
    opts  = opts or dict()
    _mode = opts.get("mode", "completion_time")
    _prec = opts.get("round", 10)

    # Шаг 2. Определить J_I, J_F
    for j, _ in enumerate(x):
        is_float = False
        min_t = None
        sol_t = []
        pj = len(x[j])
        for k, _ in enumerate(x[j]):
            for t, _ in enumerate(x[j][k]):
                x_jkt = round(x[j][k][t], _prec)
                if 0 < x_jkt and x_jkt < 1:
                    is_float = True
                if x_jkt:
                    sol_t.append(t)
                if (x_jkt > 0) and (k + 1 == pj) and (min_t is None):
                    min_t = t
        if is_float:
            jobs_float[j] = min_t
        else:
            jobs_integer[j] = sol_t
    
    # Шаг 3. Распланировать J_I как в решении
    for j in jobs_integer:
        for t in jobs_integer[j]:
            schedule[t] = j + 1

    # Шаги 4 и 5.
    def _fsort(a, b):
        j1, j2, t1, t2 = a[0], b[0], a[1], b[1]
        if t1 < t2:
            return -1
        elif t1 == t2:
            return -1 if j1 < j2 else 1
        else:
            return 1
    
    jobs_float_sorted = sorted(
        jobs_float.items(),
        key=cmp_to_key(_fsort),
        reverse=False
    )
    
    _pending_jobs = [];
    
    for j, max_t in jobs_float_sorted:
        sj = []
        for t in range(params["r"][j] - 1, max_t + 1):
            if schedule[t] < 0:
                sj.append(t)
            if len(sj) == params["p"][j]:
                break
        if len(sj) < params["p"][j]:
            _pending_jobs.append(j)
        else:
            for t in sj:
                schedule[t] = j + 1

    for j in _pending_jobs:
        rj = params["r"][j] - 1
        pj = params["p"][j]
        for t in range(rj, len(schedule)):
            if schedule[t] < 0:
                schedule[t] = j + 1
                pj -= 1
            if not pj:
                break

    return schedule

# ----------------
# test
# ----------------
# def main():
#     x = [
#         [
#             [1,  0,  0,  0,  0,  0,  0,  0],
#             [0, .5,  0,  0,  0,  0,  0, .5],
#         ],
#         [
#             [0,  0,  0,  0, .5,  0, .5,  0],
#             [0,  0,  0,  0,  0, .5,  0, .5],
#         ],
#         [
#             [0,  0, .5, .5,  0,  0,  0,  0],
#             [0,  0,  0, .5, .5,  0,  0,  0],
#         ],
#         [
#             [0, .5,  0,  0,  0, .5,  0,  0],
#             [0,  0, .5,  0,  0,  0, .5,  0],
#         ],
#     ]
#     w = []
#     params = {
#         "r": [1, 4,  3, 2],f
#         "w": [4, 9, 12, 9],
#         "p": [2, 2,  2, 2],
#     }
#     # 14334221
#     return (
#         solve_a1(w, x, params),
#         solve_a1(w, x, params, { "mode": "starting_time" })
#     )
# main()