#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <sstream>
#include <vector>
#include <tuple>
#include <limits>
#include <utility>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <stdexcept>
#include "./lapjv/lap.cpp"

typedef std::vector<int> TCostRow;
typedef std::vector<TCostRow> TCostMatrix;

typedef std::vector<int> TSolRow;
typedef std::vector<TSolRow> TSolMatrix;

typedef std::vector<int> TSolVec;

typedef std::vector<int> TArc;
typedef std::map<TArc, int> TAncestors;

typedef std::map<std::string, std::vector<int>> TJobParams;

TCostMatrix _copy_matrix(TCostMatrix &matrix) {
    int h = matrix.size();
    TCostMatrix result(h, TCostRow(h, 0));
    for (int i = 0; i < matrix.size(); i++) {
        result[i] = matrix[i];
    }
    return result;
}

int _get_matrix_max_value(TCostMatrix &w) {
    int max = -1;
    int dim = w.size();
    for (int c = 0; c < dim; c++) {
        for (int v = 0; v < dim; v++) {
            if (w[c][v] > max) {
                max = w[c][v];
            }
        }
    }
    return max;
}

int _get_digit_count(int num) {
    if (num == 0) return 1;
    num = abs(num);
    return std::floor(std::log10(num)) + 1;
};

/**
 * Напечатать матрицу
 */
void print_osp_matrix(TCostMatrix &w) {
    int dim = w.size();
    
    int maxval = _get_matrix_max_value(w);
    int maxlen = _get_digit_count(maxval);

    for (int c = 0; c < dim; c++) {
        for (int v = 0; v < dim; v++) {
            int curlen = _get_digit_count(w[c][v]);
            std::cout << std::string(maxlen + 2 - curlen, ' ') << w[c][v];
        }
        std::cout << std::endl;
    }
};

/**
 * Вернуть вектор решения как матрицу
 */
TSolMatrix get_solution_as_matrix(TSolVec &x) {
    int d = x.size();
    TSolMatrix x_sol_mat(d, TSolRow(d, 0));
    for (int c = 0; c < d; c++) {
        x_sol_mat[c][x[c]] = 1;
    }
    return x_sol_mat;
};

/**
 * Напечатать решение как матрицу
 */
void print_solution_as_matrix(TSolVec &x) {
    TSolMatrix xmat = get_solution_as_matrix(x);
    print_osp_matrix(xmat);
}

const int POS_INF = std::numeric_limits<int>::max();
const int NEG_INF = std::numeric_limits<int>::min();

/**
 * Решатель
 */
class solver_osp_lap_bnb_2d_v4_0 {
    const int POS_INF = std::numeric_limits<int>::max();
    const int NEG_INF = std::numeric_limits<int>::min();

  public:
    // Ограничить работу алгоритма указанным количеством вершин
    int n_tree_nodes_stop_limit = POS_INF;

    // Ограничить работу алгоритма указанным временем выполнения
    long double t_stop_limit = POS_INF; // sec

    std::string stop_reason = "";

    // Время запуска
    std::clock_t c_start = 0;

    // Общее время выполнения
    long double exec_time_ms = 0;

    // Количество решенных ЗН
    int n_lap_solved   = 0;

    // Количество вершин в ДПОР
    int n_tree_nodes   = 0;

    int n_cut_obj_ub   = 0;
    int n_cut_lb_ub    = 0;
    int n_cut_inf      = 0;
    int n_cut_feasible = 0;
    int n_cut_no_opt   = 0;

    // Количество созданных задач (количество созданных матриц стоимостей)
    int n_created_w    = 0;

    // Количество двоичных ветвлений
    int n_branching    = 0;

    bool reuse_parent_x_for_exclusion = true;
    bool reuse_parent_x_for_inclusion = true;

    // Усиление оценки сверху через сортировку недопустимых операций
    bool use_swap_patch = false;
    
    // Правило ветвления по дугам (0 == не задано) [обязательное поле]
    int arc_branch_rule = 0;

    // Правило ветвления по допускам (0 == отключено)
    int tolerance_branch_rule = 0;

    static const int NONE = 0;

    static const int LOWEST_UPPER_TOLERANCE = 2;
    static const int ARC_UPPER_TOLERANCE    = 4;

    static const int LOWEST_COST_ARC        = 8;
    static const int HIGHEST_COST_ARC       = 16;
    static const int FIRST_AVAILABLE_ARC    = 32;

    // Оценка сверху
    int ub = std::numeric_limits<int>::max();

    // Локальное оптимальное решение (оценка сверху)
    TSolVec x_opt;

    // Матрица стоимостей для которой решается задача
    TCostMatrix cost_matrix;

    // Характеристики работ
    TJobParams params;

    /*
     * Вернуть из матрицы стоимостей достаточно большое число (выполняет роль бесконечности)
     * Работает только для p > 1
     */
    int get_weight_inf_value(TCostMatrix &w) {
        return w[1][0];
    }

    int calc_obj(TCostMatrix &w, TSolVec &x) {
        int dim = x.size();
        int obj = 0;
        for (int c = 0; c < dim; c++) {
            obj += w[c][x[c]];
        }
        return obj;
    }

    /*
     * Исправить недопустимое решение через сортировку недопустимых операций (дуг)
     */
    void patch_swap_infeas(TSolVec &x) {
        int d = x.size();

        for (int c = 0; c < d; c++) {
            std::vector<TArc> arcs = this->get_schedule_infeasible_arcs(x);

            if (arcs[0][0] == -1) {
                break;
            }

            int i1 = arcs[0][0];
            int j1 = arcs[0][1];

            int i2 = arcs[1][0];
            int j2 = arcs[1][1];

            x[i1] = j2;
            x[i2] = j1;
        }
    }

    /*
     * Рассчитать верхний допуск
     */
    int get_lap_tolerance(
        TCostMatrix &w,
        TSolVec &x,
        TArc &arc
    ) {
        int obj = this->calc_obj(w, x);
        
        this->exclude_weight_2d(w, arc);
        
        x = this->solve_lap(w);
        
        int _obj = this->calc_obj(w, x);
        
        return _obj - obj;
    }

    /*
     * Вернуть пару дуг из недопустимого решения
     */
    std::vector<TArc> get_schedule_infeasible_arcs(TSolVec &x) {
        int h = x.size();
        int k = 0;
        int j = 0;
        int pj = this->params.at("p")[j];
        for (int a = 0; a < h; a++) {
            if (k && x[a] - x[a-1] <= 0) {
                return {
                    { a, x[a] },
                    { a - 1, x[a - 1] }
                };
            }
            pj = pj - 1;
            if (!pj) {
                j = j + 1;
                k = 0;
                pj = this->params.at("p")[j];
            } else {
                k = k + 1;
            }
        }
        return {
            { -1, -1 },
            { -1, -1 }
        };
    }

    /*
     * Принудительно включить коэф. в решение
     */
    void include_weight_2d(TCostMatrix &w, TArc arc) {
        int _inf = this->get_weight_inf_value(w);
        int i = arc[0];
        int j = arc[1];
        int v = w[i][j];
        int T = w.size();
        for (int jj = 0; jj < T; jj++) {
            w[i][jj] = _inf;
        }
        w[i][j] = v;
    }

    /*
     * Принудительно исключить коэф. из решения
     */
    void exclude_weight_2d(TCostMatrix &w, TArc arc) {
        int _inf = this->get_weight_inf_value(w);
        w[arc[0]][arc[1]] = _inf;
    }

    /*
     * Решить задачу о назначениях (алгоритм Йонкера-Фольгенанта)
     */
    TSolVec solve_lap(TCostMatrix &w) {
        // https://github.com/yongyanghz/LAPJV-algorithm-c
        // https://github.com/hrldcpr/pyLAPJV

        int dim = w.size();

        cost ** costMatrix;
        col *rowsol;
        row *colsol;
        cost *u;
        cost *v;
        rowsol = new col[dim];
        colsol = new row[dim];
        u = new cost[dim];
        v = new cost[dim];
        costMatrix = new cost*[dim];

        for (int i = 0; i < dim; i++) {
            costMatrix[i]  =  new cost[dim];
        }

        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                costMatrix[i][j] = w[i][j];
            }
        }
        
        cost totalCost = lap(dim, costMatrix, rowsol, colsol, u, v);

        TSolVec x(dim, 0);

        for (int j = 0; j < dim; ++j) {
            x[j] = rowsol[j];
            
            // утечка памяти
            delete[] costMatrix[j];
        };

        // утечка памяти
        delete[] u;
        delete[] v;
        delete[] rowsol;
        delete[] colsol;
        delete[] costMatrix;

        return x;
    }

    /*
     * Вычислить оптимальное расписание для заданной матрицы стоимостей и характеристик работ
     */
    TSolVec solve(TCostMatrix &w) {
        TAncestors _ancestors;
        TSolVec _cache_x_lap;

        this->cost_matrix = w;

        this->c_start = std::clock();

        TSolVec x = this->_solve(w, false, _ancestors, _cache_x_lap);
        
        std::clock_t c_end = std::clock();
        
        this->exec_time_ms = 1000.0 * (c_end - this->c_start) / CLOCKS_PER_SEC;

        return x;
    }

    TSolVec _solve(
        TCostMatrix &w,
        bool _is_exclusion,
        TAncestors &_ancestors,
        TSolVec &_cache_x_lap
    ) {
        // --------------------

        std::vector<TArc> infeasible_arcs;
        
        TArc branch_arc;
        TAncestors _ancestors2;
        
        TSolVec x_lap;
        TSolVec x_lap_1;
        TSolVec x_osp_1;
        TSolVec x_osp_2;
        
        TCostMatrix w1;
        TCostMatrix w2;

        int obj_1;
        int obj_2;
        int obj;
        int lb;

        int _inf;

        // --------------------

        _inf = this->get_weight_inf_value(w);

        if (
            !_is_exclusion // => is_inclusion = 1
            && this->reuse_parent_x_for_inclusion
            && this->n_tree_nodes > 1
        ) {
            x_lap = _cache_x_lap;
        }

        else if (
            _is_exclusion
            && this->reuse_parent_x_for_exclusion
            && this->tolerance_branch_rule
            && this->n_tree_nodes > 1
        ) {
            x_lap = _cache_x_lap;
        }

        else {
            x_lap = this->solve_lap(w);

            this->n_lap_solved += 1;
        }

        this->n_tree_nodes += 1;

        obj = this->calc_obj(w, x_lap);
        lb  = obj;

        // отладка
        // std::cout << "obj=" << obj << "; ub=" << this->ub <<  std::endl;
        // print_solution_as_matrix(x_lap);
        // std::cout << std::endl;

        if (this->n_tree_nodes % 10000 == 0) {
            // отладка
            // std::cout << this->n_tree_nodes << ", lb=" << lb << ", ub=" << this->ub << std::endl;

            std::clock_t c_end = std::clock();
            this->exec_time_ms = 1000.0 * (c_end - this->c_start) / CLOCKS_PER_SEC;

            if (  this->exec_time_ms >= this->t_stop_limit  ) {
                this->stop_reason = "t_stop_limit";
                throw std::runtime_error("t_stop_limit");
            }
        }

        if (  this->n_tree_nodes >= this->n_tree_nodes_stop_limit  ) {
            this->stop_reason = "n_tree_nodes_stop_limit";
            throw std::runtime_error("n_tree_nodes_stop_limit");
        }

        if (lb > this->ub) {
            this->n_cut_obj_ub += 1;
            this->n_cut_lb_ub += 1;

            return this->x_opt;
        }

        if (lb >= _inf) {
            this->n_cut_inf += 1;

            return this->x_opt;
        }

        infeasible_arcs = this->get_schedule_infeasible_arcs(x_lap);

        if (infeasible_arcs[0][0] == -1) {
            this->n_cut_feasible += 1;

            this->x_opt = x_lap;
            this->ub = obj;

            return this->x_opt;
        }

        branch_arc = { -1, -1 }; // = undefined arc

        if (
               (LOWEST_COST_ARC == this->arc_branch_rule)
            || (HIGHEST_COST_ARC == this->arc_branch_rule)
        ) {
            for ( auto arc : infeasible_arcs ) {
                auto it = _ancestors.find(arc);
                if (it != _ancestors.end()) {
                    // if arc in _ancestors:
                    continue;
                }
                if ( branch_arc[0] == -1 ) {
                    branch_arc = arc;
                    continue;
                }

                int ci = arc[0];
                int cj = arc[1];
                int mi = branch_arc[0];
                int mj = branch_arc[1];

                if (LOWEST_COST_ARC == this->arc_branch_rule) {
                    if (w[ci][cj] < w[mi][mj]) {
                        branch_arc = arc;
                    }
                } else {
                    if (w[ci][cj] > w[mi][mj]) {
                        branch_arc = arc;
                    }
                }
            }
        }
        else if (FIRST_AVAILABLE_ARC == this->arc_branch_rule) {
            for ( auto arc : infeasible_arcs ) {
                auto it = _ancestors.find(arc);
                if (it != _ancestors.end()) {
                    // if arc in _ancestors:
                    continue;
                }
                if ( branch_arc[0] == -1 ) {
                    branch_arc = arc;
                    break;
                }
            }
        }
        else {
            throw std::invalid_argument("unknown arc_branch_rule");
        }

        // out of options
        if (branch_arc[0] == -1) {
            this->n_cut_no_opt += 1;

            return this->x_opt;
        }

        TSolVec _cur_x_lap;
        TCostMatrix _cur_w;
        int _cur_tol;

        if (this->tolerance_branch_rule) {
            int branch_tol = this->POS_INF;

            if (LOWEST_UPPER_TOLERANCE == this->tolerance_branch_rule) {
                branch_arc = { -1, -1 };

                for ( auto arc : infeasible_arcs ) {
                    auto it = _ancestors.find(arc);
                    if (it != _ancestors.end()) {
                        // if arc in _ancestors:
                        continue;
                    }

                    _cur_x_lap = x_lap;
                    _cur_w     = _copy_matrix(w);
                    _cur_tol   = this->get_lap_tolerance(_cur_w, _cur_x_lap, arc);

                    this->n_created_w  += 1;
                    this->n_lap_solved += 1;

                    if (  (branch_arc[0] == -1) || (_cur_tol < branch_tol)  ) {
                        branch_arc = arc;
                        branch_tol = _cur_tol;
                        w1         = _cur_w;
                        x_lap_1    = _cur_x_lap;
                    }
                }
            }

            else if (ARC_UPPER_TOLERANCE == this->tolerance_branch_rule) {
                x_lap_1    = x_lap;
                w1         = _copy_matrix(w);
                branch_tol = this->get_lap_tolerance(w1, x_lap_1, branch_arc);

                this->n_created_w  += 1;
                this->n_lap_solved += 1;
            }

            else {
                throw std::invalid_argument("unknown tolerance_branch_rule");
            }

            lb = obj + branch_tol;

            if ((lb < this->ub) == false) {
                this->n_cut_lb_ub += 1;

                return this->x_opt;
            }
        }

        if (this->use_swap_patch) {
            _cur_x_lap = x_lap;
            
            this->patch_swap_infeas(_cur_x_lap);
            
            std::vector<TArc> _infeasible_arcs = this->get_schedule_infeasible_arcs(_cur_x_lap);

            if (_infeasible_arcs[0][0] == -1) { // => is_valid_schedule
                int _obj = this->calc_obj(w, _cur_x_lap);

                if (_obj < this->ub) {
                    this->ub = _obj;
                    this->x_opt = _cur_x_lap;
                }

                if ((lb < this->ub) == false) {
                    this->n_cut_lb_ub += 1;

                    return this->x_opt;
                }
            }
        }

        // очистить память
        _cur_x_lap.clear();
        _cur_x_lap.shrink_to_fit();
        _cur_w.clear();
        _cur_w.shrink_to_fit();

        if (!this->tolerance_branch_rule) {
            w1 = _copy_matrix(w);
            this->exclude_weight_2d(w1, branch_arc);
            this->n_created_w += 1;
        }

        w2 = _copy_matrix(w);
        this->include_weight_2d(w2, branch_arc);
        this->n_created_w += 1;

        _ancestors2 = _ancestors;
        _ancestors2[branch_arc] = 1;

        this->n_branching += 1;

        x_osp_1 = this->_solve(
            w1,
            true, // is_exclusion=1
            _ancestors2,
            x_lap_1
        );

        obj_1 = this->calc_obj(w1, x_osp_1);

        // очистить память
        w1.clear();
        w1.shrink_to_fit();
        x_lap_1.clear();
        x_lap_1.shrink_to_fit();

        x_osp_2 = this->_solve(
            w2,
            false, // is_inclusion=1
            _ancestors2,
            x_lap
        );

        obj_2 = this->calc_obj(w2, x_osp_2);

		// очистить память
        w2.clear();
        w2.shrink_to_fit();
        x_lap.clear();
        x_lap.shrink_to_fit();

        if (obj_1 < obj_2) {
            return x_osp_1;
        } else {
            return x_osp_2;
        }
    }
};

/*
 * Грубая оценка сверху для инициализации
 */
void get_eager_upper_bound(
    TCostMatrix &w,
    TSolVec &x,
    int &ub
) {
    int _inf = w[1][0];
    int h    = w.size();

    std::vector<int> s(h, 0);

    for (int a = 0; a < h; a++) {
        for (int t = 0; t < h; t++) {
            if (w[a][t] < _inf && !s[t]) {
                s[t] = 1;
                ub += w[a][t];
                x[a] = t;
                break;
            }
        }
    }
};

/*
 * Вывести результаты как JSON
 */
void print_response_json(solver_osp_lap_bnb_2d_v4_0 solver) {
    std::stringstream res;
    std::stringstream x_str;

    int dim = solver.x_opt.size();
    int obj = solver.calc_obj(solver.cost_matrix, solver.x_opt);

    x_str << "[";
    for (int c = 0; c < dim; c++) {
        if (c) x_str << ", ";
        x_str << solver.x_opt[c];
    }
    x_str << "]";

    res << "{"
        << "\"obj\": " << obj
        << ", \"x\": " << x_str.str()
        << ", \"stop_reason\": \""  << solver.stop_reason << "\""
        << ", \"exec_time_ms\": "   << solver.exec_time_ms
        << ", \"n_tree_nodes\": "   << solver.n_tree_nodes
        << ", \"n_lap_solved\": "   << solver.n_lap_solved
        << ", \"n_cut_obj_ub\": "   << solver.n_cut_obj_ub
        << ", \"n_cut_lb_ub\": "    << solver.n_cut_lb_ub
        << ", \"n_cut_inf\": "      << solver.n_cut_inf
        << ", \"n_cut_feasible\": " << solver.n_cut_feasible
        << ", \"n_cut_no_opt\": "   << solver.n_cut_no_opt
        << ", \"n_created_w\": "    << solver.n_created_w
        << ", \"n_branching\": "    << solver.n_branching
        << "}";

    std::cout << res.str() << std::endl;
}

/*
 * Вывести результаты как текст
 */
void print_response_text(solver_osp_lap_bnb_2d_v4_0 solver) {
    int dim = solver.x_opt.size();

    std::cout << "solver.x_opt = ";
    for (int i = 0; i < dim; i++) {
        std::cout << "(" << i << "," << solver.x_opt[i] << ") ";
    }
    std::cout << std::endl;

    int obj = solver.calc_obj(solver.cost_matrix, solver.x_opt);

    std::cout << "obj = " << obj << std::endl;
    std::cout << "exec_time_ms = "   << solver.exec_time_ms      << " ms" << std::endl;
    std::cout << "n_tree_nodes = "   << solver.n_tree_nodes      << std::endl;
    std::cout << "n_lap_solved = "   << solver.n_lap_solved      << std::endl;

    std::cout << "n_cut_obj_ub = "   << solver.n_cut_obj_ub      << std::endl;
    std::cout << "n_cut_lb_ub = "    << solver.n_cut_lb_ub       << std::endl;
    std::cout << "n_cut_inf = "      << solver.n_cut_inf         << std::endl;
    std::cout << "n_cut_feasible = " << solver.n_cut_feasible    << std::endl;
    std::cout << "n_cut_no_opt = "   << solver.n_cut_no_opt      << std::endl;
    std::cout << "n_created_w = "    << solver.n_created_w       << std::endl;
    std::cout << "n_branching = "    << solver.n_branching       << std::endl;
}

/*
 * Прочитать строку из чисел и записать в массив
 */
std::vector<int> stringToIntVector(const std::string& input) {
    std::vector<int> result;
    std::stringstream ss(input);
    int num;
    while (ss >> num) {
        result.push_back(num);
    }
    return result;
}

int main(int argc, char** argv) {
    std::string inp_src = argv[argc - 1];
    // std::string out_dst = argv[argc - 1];

    // Чтение задачи из файла
    // -------------------------------
    std::ifstream file(inp_src);

    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        return 1;
    }

    std::string line;

    // количество работ
    // std::getline(file, line);
    // int n = stoi(line);

    // характеристики работ
    TJobParams params;
    std::vector<std::string> _param_keys = { "p", "w", "r", "s1", "s2", "d1", "d2" };
    for (int c = 0; c < 7; c++) {
        std::getline(file, line);
        params[_param_keys[c]] = stringToIntVector(line);
    }

    // матрица стоимостей
    TCostMatrix w;
    while (std::getline(file, line)) {
        w.push_back(
            stringToIntVector(line)
        );
    }

    // print_osp_matrix(w);

    file.close();


    // Решатель
    // -------------------------------
    solver_osp_lap_bnb_2d_v4_0 solver;

    solver.params = params;

    // Инициализации оценки сверху
    int obj_ub = 0;
    TSolVec x_ub(w.size(), 0);
    get_eager_upper_bound(w, x_ub, obj_ub);
    solver.x_opt = x_ub;
    solver.ub = obj_ub;

    // Верхняя оценка сортировкой
    solver.use_swap_patch = true;

    // Допуски
    solver.tolerance_branch_rule = solver_osp_lap_bnb_2d_v4_0::NONE;
    // solver.tolerance_branch_rule = solver_osp_lap_bnb_2d_v4_0::ARC_UPPER_TOLERANCE;
    // solver.tolerance_branch_rule = solver_osp_lap_bnb_2d_v4_0::LOWEST_UPPER_TOLERANCE;

    // Правило ветвления
    // solver.arc_branch_rule = solver_osp_lap_bnb_2d_v4_0::LOWEST_COST_ARC
    // solver.arc_branch_rule = solver_osp_lap_bnb_2d_v4_0::HIGHEST_COST_ARC
    // solver.arc_branch_rule = solver_osp_lap_bnb_2d_v4_0::FIRST_AVAILABLE_ARC

    bool _print_sol_matrix = false;

    for (int c = 0; c < argc; c++) {
        std::string arg = argv[c];

        if ("--arc_upper_tolerance" == arg) {
            solver.tolerance_branch_rule = solver_osp_lap_bnb_2d_v4_0::ARC_UPPER_TOLERANCE;
        }
        else if ("--lowest_upper_tolerance" == arg) {
            solver.tolerance_branch_rule = solver_osp_lap_bnb_2d_v4_0::LOWEST_UPPER_TOLERANCE;
        }
        else if ("--lowest_cost_arc" == arg) {
            solver.arc_branch_rule = solver_osp_lap_bnb_2d_v4_0::LOWEST_COST_ARC;
        }
        else if ("--highest_cost_arc" == arg) {
            solver.arc_branch_rule = solver_osp_lap_bnb_2d_v4_0::HIGHEST_COST_ARC;
        }
        else if ("--first_available_arc" == arg) {
            solver.arc_branch_rule = solver_osp_lap_bnb_2d_v4_0::FIRST_AVAILABLE_ARC;
        }
        else if ("--print_solution_matrix" == arg) {
            _print_sol_matrix = true;
        }
    }

    solver.t_stop_limit = 1000 * 60 * 90; // 90 min = 1.5 hours
    // solver.n_tree_nodes_stop_limit = 1000;

    try {
        solver.solve(w);
    } catch (const std::exception& e) {
        // std::cout << e.what() << std::endl;
    }

    // print_response_text(solver);
    // std::cout << std::endl;
    
    print_response_json(solver);

    if (_print_sol_matrix) {
        std::cout << std::endl;
        print_solution_as_matrix(solver.x_opt);
    }

    return 0;
}