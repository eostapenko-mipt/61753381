import os
import json
import random
import asyncio
import subprocess
import aiofiles
import aiofiles.os
import time

import lib.config

# Обработать ответ из решателя
def process_solver_output(x, p):
    # x: list[j,p,t] = ответ из решателя
    # p: list        = количество операций в работах
    
    sol = []
    for j, pj in enumerate(p):
        sol_job = []
        for k in range(pj):
            sol_job.append(x[j][k])
        sol.append(sol_job)
    return sol


# Подготовить и вернуть данные для решателя
def create_solver_data_str(w):
    # w: list[j,k,t] = матрица взвешенного времени

    _inf   = w[0][1][0] # достаточно большое число (выполняет роль бесконечности) # _inf = np.max(w)
    n      = len(w)     # количество работ
    p      = []         # количество операций
    max_p  = 0          # максимальное количество операций из представленных
    T      = 0          # временной горизонт

    for _w in w:
        pj    = len(_w)
        max_p = max(pj, max_p)
        T     += pj
        p.append(pj)

    res = ""
    
    for j, _ in enumerate(w):
        for k, _ in enumerate(w[j]):
            for w_jkt in w[j][k]:
                res += str(int(w_jkt)) + ","
            res += "\n"
        for k in range(  max_p - len(w[j])  ):
            res += ",".join(["D"] * T) + ","
            res += "\n"
        res += "\n"
        
    res = res.strip("\n").strip(",")
    
    data = ""
    data += f"INF = {_inf};\n"
    data += f"N = {n};\n"
    data += f"PROC_TIMES = [{','.join(map(str, p))}];\n\n"
    data += f"ln_weights = array3d(JOB_INDICES, JOB_PART_INDICES, TIME_INTERVAL_INDICES, [\n{res}\n]);\n"
    
    return data


# Решить задачу ЛП
async def solve_lp(
    w,
    model_path,
    data_path=None,
    ctx=None
):
    # w: list[j,k,t]  = матрица взвешенного времени
    # model_path: str = путь к модели

    data_dzn = create_solver_data_str(w)

    should_delete_data_file = False

    if not data_path:
        should_delete_data_file = True
        data_path = lib.config.global_config["default_minizinc_data_dir"] + "/" + str(random.random())[3:] + ".dzn"

    async with aiofiles.open(data_path, "w") as fd:
        await fd.write(data_dzn)
    
    cmd = (lib.config.global_config["minizinc_exec_path"]
    + ' --solver "HiGHS"'
    + f' --model "{model_path}"'
    + f' --data "{data_path}"'
    + ' --output-mode json'
    # + ' --output-time'
    + ' --solution-separator " "'
    + ' --search-complete-msg " "'
    + ' --solution-comma " "')

    t0 = time.time()
    
    # proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    # (out, err) = proc.communicate()
    out, err = await proc.communicate()
    
    t1 = time.time()

    if err:
        print(err)

    p = [len(w_) for w_ in w]

    out_parsed = json.loads(out)

    if ctx is not None:
        ctx.update(out_parsed)
        ctx["exec_time"] = t1 - t0
    
    x = out_parsed["ln_activators"]
    x = process_solver_output(x, p)

    if should_delete_data_file:
        await aiofiles.os.remove(data_path)

    return x


# Решить задачу составления оптимального расписания
async def solve_osp_blp(w, ctx=None):
    # w: list[j,k,t] = матрица взвешенного времени

    return await solve_lp(w=w, model_path="./model_blp.mzn", ctx=ctx)