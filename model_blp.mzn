include "globals.mzn";

int: INF;  % Относительно большое число -- вместо бесконечности
int: N;    % Количество работ
int: M = INF;
int: D = INF;

set of int: JOB_INDICES = 1..N;                       % Индексы работ

array[JOB_INDICES] of int: PROC_TIMES;                % Количество задач в работах

set of int: JOB_PART_INDICES = 1..max(PROC_TIMES);    % Индексы задач

int: T = sum(PROC_TIMES); % Временной горизонт

set of int: TIME_INTERVAL_INDICES = 1..T; % Моменты времени. От 1 и до горизонта T

% Лин. коэф. Активаторы
array[JOB_INDICES, JOB_PART_INDICES, TIME_INTERVAL_INDICES] of var 0..1: ln_activators;

% Лин. коэф. Веса
array[JOB_INDICES, JOB_PART_INDICES, TIME_INTERVAL_INDICES] of int: ln_weights;

% Ограничение. Одна работа в один момент времени "t"
constraint
    forall(j in JOB_INDICES)(
        forall(k in 1..PROC_TIMES[j])(
            sum(t in TIME_INTERVAL_INDICES)(
                ln_activators[j,k,t]
            ) == 1
        )
    );

% Ограничение. Одна задача в один момент времени "t"
constraint
    forall(t in TIME_INTERVAL_INDICES)(
        sum(j in JOB_INDICES)(
            sum(k in 1..PROC_TIMES[j])(
                ln_activators[j,k,t]
            )
        ) == 1
    );

% Вспомогательная таблица для ограничения порядка
array[JOB_INDICES, JOB_PART_INDICES, TIME_INTERVAL_INDICES] of var int: ln_order;
constraint
    forall(j in JOB_INDICES)(
        forall(k in JOB_PART_INDICES)(
            forall(t in TIME_INTERVAL_INDICES)(
                ln_order[j, k, t] = t
            )
        )
    );

% Ограничение. Порядок выполнения задач: j.k_i < j.k_{i+1}
% т.е. "Задача 3" не может быть выполнена раньше "Задачи 2"
constraint
    forall(j in JOB_INDICES)(
        forall(k in JOB_PART_INDICES where k > 1)(
            sum(t in TIME_INTERVAL_INDICES)(
                ln_order[j, k, t] * ln_activators[j,k,t]
            )
            - sum(t in TIME_INTERVAL_INDICES)(
                ln_order[j, k-1, t] * ln_activators[j, k-1 ,t]
            )
            > 0
        )
    );

% Целевая функция для минимизации
var int: target_value = (
    sum(j in JOB_INDICES)(
        sum(k in 1..PROC_TIMES[j])(
            sum(t in TIME_INTERVAL_INDICES)(
                ln_weights[j,k,t] * ln_activators[j,k,t]
            )
        )
    )
);

% Найти минимум целевой функции
solve minimize target_value;

% output ["target_value = "] ++ [show(target_value)] ++ ["\n\n"];

% output ["activation_matrix = "] ++ ["\n"];
% output [show(ln_activators[1, k, TIME_INTERVAL_INDICES]) ++ "\n" | k in 1..PROC_TIMES[1]];
% output [show(ln_activators[2, k, TIME_INTERVAL_INDICES]) ++ "\n" | k in 1..PROC_TIMES[2]];
% output [show(ln_activators[3, k, TIME_INTERVAL_INDICES]) ++ "\n" | k in 1..PROC_TIMES[3]] ++ ["\n"];

% output [show(ln_activators)]



