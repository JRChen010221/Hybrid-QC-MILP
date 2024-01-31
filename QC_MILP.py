import os
import json
import itertools
import time
import numpy as np
import pyomo.environ as pyo
from typing import List
from collections import namedtuple
from pyomo.opt import SolverStatus, TerminationCondition
from dimod import BinaryQuadraticModel
from dwave.system import EmbeddingComposite, DWaveSampler
from dwave.samplers import SimulatedAnnealingSampler
from data import JobshopData


class QCMILPSolver:
    def __init__(self, job_num: int, machine_num: int, data: JobshopData, device: str) -> None:
        self.job_num = job_num
        self.machine_num = machine_num
        self.data = data
        self.device = device
        self.pyo_model = None

    def solve(self, save_pth: str) -> None:
        """
        Solve the scheduling and sequencing problems for each instance in the data.
        
        Args:
            save_pth (str): The path to save the results.
        
        Returns:
            None
        """
        solve_time = []
        optimal_xs = []
        optimal_ts = []
        feasible_sequence = []
        optimal_values = []
        for i, instance in enumerate(self.data.instances):
            p_arr, c_arr, r_arr, d_arr = instance
            self._construct_model(p_arr, c_arr, r_arr, d_arr)
            optimal_flag = False
            n_iter = 0
            total_time = 0
            print(f"************Solving instance {i+1}************")
            while not optimal_flag:
                n_iter += 1
                print(f"------------Iteration {n_iter}------------")
                optimal_flag, optimal_x, optimal_t, optimal_value, milp_time = self._solve_relaxed_milp()
                total_time += milp_time
                if not optimal_flag:
                    print(f"The scheduling problem {i+1} has no feasible solution!")
                    print(f"************End solving instance {i+1}************")
                    solve_time.append(-1)
                    optimal_xs.append(-1)
                    optimal_ts.append(-1)
                    feasible_sequence.append(-1)
                    optimal_values.append(-1)
                    break
                else:
                    optimal_flag, sequence, infeasible_m, qubo_time = self._solve_qubo(optimal_x, optimal_t, p_arr)
                    total_time += qubo_time
                    if not optimal_flag:
                        print(f"The total sequencing problem is infeasible with the following machines: {infeasible_m}")
                    else:
                        print(f"The scheduling problem {i+1} is feasible with optimal value {optimal_value} "
                              f"and solving time {total_time}")
                        print(f"************End solving instance {i+1}************")
                        solve_time.append(total_time)
                        optimal_xs.append(optimal_x.tolist())
                        optimal_ts.append(optimal_t.tolist())
                        feasible_sequence.append(sequence.tolist())
                        optimal_values.append(optimal_value)
        result_dict = {}
        result_dict["instance_num"] = len(self.data.instances)
        result_dict["solve_time"] = solve_time
        result_dict["mean_time"] = np.mean(solve_time)
        result_dict["std_time"] = np.std(solve_time)
        result_dict["optimal_xs"] = optimal_xs
        result_dict["optimal_ts"] = optimal_ts
        result_dict["feasible_sequence"] = feasible_sequence
        result_dict["optimal_values"] = optimal_values
        result_dict["mean_values"] = np.mean(optimal_values)
        result_dict["std_values"] = np.std(optimal_values)
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)
        result_pth = os.path.join(save_pth, f"QC_{self.job_num}_{self.machine_num}.json")
        with open(result_pth, 'w') as wf:
            json.dump(result_dict, wf)

    def _construct_model(self, p_arr: np.ndarray, c_arr: np.ndarray, 
                           r_arr: np.ndarray, d_arr: np.ndarray) -> None:
        """
        Constructs a Pyomo model using the provided arrays for process time, process cost, release time, and due time.
        
        Parameters:
            p_arr (np.ndarray): The array of process times.
            c_arr (np.ndarray): The array of process costs.
            r_arr (np.ndarray): The array of release times.
            d_arr (np.ndarray): The array of due times.
        
        Returns:
            None
        """
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
        model.objective = pyo.Objective(expr=sum(model.process_cost[j, m] * model.x[j, m] 
                                                 for j in model.j_index for m in model.m_index))
        model.constraints = pyo.ConstraintList()
        for j in model.j_index:
            model.constraints.add(model.t[j] >= model.release_time[j])
            model.constraints.add(model.t[j] + sum(model.process_time[j, m] * model.x[j, m] for m in model.m_index) 
                                  <= model.due_time[j])
            model.constraints.add(sum(model.x[j, m] for m in model.m_index) == 1)
        self.pyo_model = model

    def _solve_relaxed_milp(self) -> tuple[bool, np.ndarray | None, np.ndarray | None, float | None, float | None]:
        """
        Solve the relaxed MILP problem and return the feasibility, optimal solution,
        solve time, and other related information.

        Returns:
            tuple[bool, np.ndarray | None, np.ndarray | None, float | None, float | None]: 
            A tuple containing the feasibility status, optimal solution for x, optimal 
            solution for t, optimal value, and solve time.
        """
        solver = pyo.SolverFactory('gurobi', solver_io='python')
        start = time.time()
        solver_results = solver.solve(self.pyo_model)
        end = time.time()
        feasibility = False
        if (solver_results.solver.status == SolverStatus.ok) and \
           (solver_results.solver.termination_condition == TerminationCondition.optimal):
            feasibility = True
            optimal_x = np.array([[pyo.value(self.pyo_model.x[j, m]) for m in self.pyo_model.m_index] 
                                   for j in self.pyo_model.j_index])
            optimal_t = np.array([pyo.value(self.pyo_model.t[j]) for j in self.pyo_model.j_index])
            optimal_value = pyo.value(self.pyo_model.objective)
            solve_time = end - start
            print(f"The relaxed MILP problem is feasible")
            return feasibility, optimal_x, optimal_t, optimal_value, solve_time
        elif solver_results.solver.status == TerminationCondition.infeasible:
            print("The relaxed MILP problem is infeasible!")
            return feasibility, None, None, None, None
        else:
            print(str(solver_results.solver))
            return feasibility, None, None, None, None
        
    def _construct_qubo(self, optimal_x: np.ndarray, optimal_t: np.ndarray, 
                        p_arr: np.ndarray, jobs_assigned: List[int]) -> tuple[BinaryQuadraticModel, List[tuple], float]:
        """
        Constructs the QUBO model based on the given relaxed MILP solution.

        Args:
            optimal_x (np.ndarray): The optimal x_{jm}.
            optimal_t (np.ndarray): The optimal t_j.
            p_arr (np.ndarray): Process time.
            jobs_assigned (List[int]): The list of assigned jobs to a machine.

        Returns:
            tuple[BinaryQuadraticModel, List[tuple], float]: A tuple containing the constructed
            BinaryQuadraticModel, the list of combinations of assigned jobs indices, and the constant value U.
        """
        bqm1 = BinaryQuadraticModel('BINARY')
        bqm2 = BinaryQuadraticModel('BINARY')
        U = sum(max(p_arr[j, :]) for j in range(p_arr.shape[0]))
        combination = list(itertools.combinations(jobs_assigned, 2))
        for i, j in combination:
            K1 = ((optimal_t[j]-optimal_t[i]-sum(p_arr[i, k]*optimal_x[i, k] 
                    for k in range(optimal_x.shape[1]))))/U
            K2 = ((optimal_t[i]-optimal_t[j]-sum(p_arr[j, k]*optimal_x[j, k] 
                    for k in range(optimal_x.shape[1]))))/U
            bqm1.add_variable(f"y_{i}_{j}", -1)
            bqm1.add_variable(f"y_{j}_{i}", -1)
            bqm1.add_interaction(f"y_{i}_{j}", f"y_{j}_{i}", 2)
            bqm2.add_variable(f"y_{i}_{j}", -K1)
            bqm2.add_variable(f"y_{j}_{i}", -K2)
        bqm_model = bqm1 + bqm2
        return bqm_model, combination, U
        
    def _solve_qubo(self, optimal_x: np.ndarray, optimal_t: np.ndarray, p_arr: np.ndarray) -> \
                    tuple[bool, np.ndarray, List[int], float]:   
        """
        Solve the QUBO problem using the given relaxed MILP solution.

        Args:
            optimal_x (np.ndarray): The optimal x_{jm}.
            optimal_t (np.ndarray): The optimal t_j.
            p_arr (np.ndarray): Process time.

        Returns:
            tuple[bool, np.ndarray, List[int], float]: A tuple containing the total
            feasibility flag, y_{ij} sequence, infeasible machine list, and the solving time.
        """
        total_feasibility = True
        sequence = np.zeros((optimal_x.shape[0], optimal_x.shape[0]))
        infeasible_m = []
        solve_time = 0
        for m in range(optimal_x.shape[1]):
            jobs_assigned = np.nonzero(optimal_x[:, m])[0]
            if len(jobs_assigned) > 1:
                bqm_model, combination, U = self._construct_qubo(optimal_x, optimal_t, p_arr, jobs_assigned)
                if self.device == "sim":
                    sampler = SimulatedAnnealingSampler()
                    sampleset = sampler.sample(bqm_model, num_reads=1000, seed=1234)
                    sa_time = (sampleset.info['timing']['preprocessing_ns'] + 
                               sampleset.info['timing']['sampling_ns'] +
                               sampleset.info['timing']['postprocessing_ns'] ) / 1e9
                    solve_time += sa_time
                else:
                    sampler = EmbeddingComposite(DWaveSampler())
                    sampleset = sampler.sample(bqm_model, num_reads=1000, annealing_time=20)
                    solve_time += (sampleset.info['timing']['qpu_access_time'] / 1e6)
                feasibility = False
                for sample in sampleset.data(fields=['sample']):
                    feasibility = self._check_qubo_feasibility(p_arr, optimal_t, m, U, sample, combination)
                    if feasibility:
                        for i, j in combination:
                            sequence[i, j] = sample[0][f"y_{i}_{j}"]
                            sequence[j, i] = sample[0][f"y_{j}_{i}"]
                        break
                if not feasibility:
                    total_feasibility = False
                    infeasible_m.append(m+1)
                    cut_expr = sum(self.pyo_model.x[j, m+1] for j in (jobs_assigned+1))
                    self.pyo_model.constraints.add(cut_expr <= (len(jobs_assigned)-1))
            else:
                continue
        return total_feasibility, sequence, infeasible_m, solve_time
    
    def _check_qubo_feasibility(self, p_arr: np.ndarray, optimal_t: np.ndarray, m: int, 
                                U: float, sample: namedtuple, combination: List[tuple]) -> bool:
        """
        Check the feasibility of a QUBO solution for a given set of parameters.

        Args:
            p_arr (np.ndarray): Process time.
            optimal_t (np.ndarray): The optimal t_j values.
            m (int): The machine index.
            U (float): Constant big value.
            sample (namedtuple): The sample data.
            combination (List[tuple]): The combination of i, j pairs for indexing y_{ij}.

        Returns:
            feasibility (bool): True if the QUBO solution is feasible, False otherwise.
        """
        feasibility = True
        for i, j in combination:
            check1 = bool(sample[0][f"y_{i}_{j}"]+sample[0][f"y_{j}_{i}"] == 1)
            check2 = bool(optimal_t[j] >= (optimal_t[i]+p_arr[i, m]-U*(1-sample[0][f"y_{i}_{j}"])))
            check3 = bool(optimal_t[i] >= (optimal_t[j]+p_arr[j, m]-U*(1-sample[0][f"y_{j}_{i}"])))
            if not (check1 and check2 and check3):
                feasibility = False
        return feasibility 
