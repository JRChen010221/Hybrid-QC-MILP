import os
import time
import json
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from data import JobshopData

class ClassicSolver:
    def __init__(self, job_num: int, machine_num: int) -> None:
        self.job_num = job_num
        self.machine_num = machine_num
        self.data = JobshopData(job_num, machine_num)

    def solve(self, save_pth: str):
        solve_time = []
        optimal_xs = []
        optimal_ts = []
        optimal_ys = []
        optimal_values = []
        for i, instance in enumerate(self.data.instances):
            p_arr, c_arr, r_arr, d_arr = instance
            model = self.construct_model(p_arr, c_arr, r_arr, d_arr)
            print(f"************Solving instance {i+1}************")
            feasibility, optimal_x, optimal_t, optimal_y, optimal_value, solve_time_i = self.solve_model(model)
            print(f"************End solving instance {i+1}************")
            if feasibility:
                optimal_xs.append(optimal_x.tolist())
                optimal_ts.append(optimal_t.tolist())
                optimal_ys.append(optimal_y.tolist())
                optimal_values.append(optimal_value)
                solve_time.append(solve_time_i)
            else:
                optimal_xs.append(-1)
                optimal_ts.append(-1)
                optimal_ys.append(-1)
                optimal_values.append(-1)
                solve_time.append(-1)
        result_dict = {}
        result_dict["instance_num"] = len(self.data.instances)
        result_dict["solve_time"] = solve_time
        result_dict["mean_time"] = np.mean(solve_time)
        result_dict["std_time"] = np.std(solve_time)
        result_dict["optimal_xs"] = optimal_xs
        result_dict["optimal_ts"] = optimal_ts
        result_dict["feasible_sequence"] = optimal_ys
        result_dict["optimal_values"] = optimal_values
        result_dict["mean_values"] = np.mean(optimal_values)
        result_dict["std_values"] = np.std(optimal_values)
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)
        result_pth = os.path.join(save_pth, f"C_{self.job_num}_{self.machine_num}.json")
        with open(result_pth, 'w') as wf:
            json.dump(result_dict, wf)
            

    def construct_model(self, p_arr: np.ndarray, c_arr: np.ndarray, 
                           r_arr: np.ndarray, d_arr: np.ndarray):
        U = sum(max(p_arr[j, :]) for j in range(self.job_num))
        model = pyo.ConcreteModel()
        model.j_index = pyo.RangeSet(1, p_arr.shape[0])
        model.m_index = pyo.RangeSet(1, p_arr.shape[1])
        model.process_time = pyo.Param(model.j_index, model.m_index, 
                                       initialize=lambda model, j, m: p_arr[j-1, m-1])
        model.process_cost = pyo.Param(model.j_index, model.m_index,
                                       initialize=lambda model, j, m: c_arr[j-1, m-1])
        model.release_time = pyo.Param(model.j_index, initialize=lambda model, j: r_arr[0, j-1])
        model.due_time = pyo.Param(model.j_index, initialize=lambda model, j: d_arr[0, j-1])
        model.x = pyo.Var(model.j_index, model.m_index, domain=pyo.Binary)
        model.t = pyo.Var(model.j_index, domain=pyo.NonNegativeReals)
        model.y = pyo.Var(model.j_index, model.j_index, domain=pyo.Binary)
        model.objective = pyo.Objective(expr=sum(model.process_cost[j, m] * model.x[j, m] 
                                                 for j in model.j_index for m in model.m_index))
        model.constraints = pyo.ConstraintList()
        for j in model.j_index:
            model.constraints.add(model.t[j] >= model.release_time[j])
            model.constraints.add(model.t[j] + sum(model.process_time[j, m] * model.x[j, m] for m in model.m_index) 
                                  <= model.due_time[j])
            model.constraints.add(sum(model.x[j, m] for m in model.m_index) == 1)
        def sequence_rule1(m, j, jp, k):
            if j <= jp:
                return m.y[j, jp] + m.y[jp, j] >= m.x[j, k] + m.x[jp, k] - 1
            else:
                return pyo.Constraint.Skip
        def sequence_rule2(m, j, jp):
            if j != jp:
                return m.t[jp] >= m.t[j] + sum(p_arr[j-1, k-1] * m.x[j, k] for k in m.m_index) - \
                    U * (1 - m.y[j, jp])
            else:
                return pyo.Constraint.Skip
        def logical_cut1(m, j, jp):
            if j < jp:
                return m.y[j, jp] + m.y[jp, j] <= 1
            else:
                return pyo.Constraint.Skip
        def logical_cut2(m, j, jp, k, kp):
            if (j < jp) and (k != kp):
                return m.y[j, jp] + m.y[jp, j] + m.x[j, k] + m.x[jp, kp] <= 2
            else:
                return pyo.Constraint.Skip
        model.sequence_constraints1 = pyo.Constraint(model.j_index, model.j_index, 
                                                     model.m_index, rule=sequence_rule1)
        model.sequence_constraint2 = pyo.Constraint(model.j_index, model.j_index, rule=sequence_rule2)
        model.logical_cut1 = pyo.Constraint(model.j_index, model.j_index, rule=logical_cut1)
        model.logical_cut2 = pyo.Constraint(model.j_index, model.j_index, 
                                            model.m_index, model.m_index, rule=logical_cut2)
        return model
    
    def solve_model(self, model):
        feasibility = False
        solver = pyo.SolverFactory('gurobi', solver_io='python')
        start = time.time()
        results = solver.solve(model, tee=True)
        end = time.time()
        solve_time = end - start
        if (results.solver.status == SolverStatus.ok) and \
           (results.solver.termination_condition == TerminationCondition.optimal):
            feasibility = True
            optimal_x = np.array([[pyo.value(model.x[j, m]) for m in model.m_index] for j in model.j_index])
            optimal_t = np.array([pyo.value(model.t[j]) for j in model.j_index])
            optimal_y = np.array([[pyo.value(model.y[i, j]) for i in model.j_index] for j in model.j_index])
            optimal_value = pyo.value(model.objective)
            print(f"The scheduling problem is feasible with optimal value {optimal_value} and solving time {solve_time}")
            return feasibility, optimal_x, optimal_t, optimal_y, optimal_value, solve_time
        elif results.solver.status == TerminationCondition.infeasible:
            print("The scheduling problem is infeasible!")
            return feasibility, None, None, None, None, None
        else:
            print(str(results.solver))
            return feasibility, None, None, None, None, None
        