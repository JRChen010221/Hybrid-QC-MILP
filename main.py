import argparse
from QC_MILP import QCMILPSolver
from classic_MILP import ClassicSolver
from data import JobshopData
from visualiaztion import plot_gantt_chart

parser = argparse.ArgumentParser()
parser.add_argument("--job_num", "-j", type=int, default=10)
parser.add_argument("--machine_num", "-m", type=int, default=4)
parser.add_argument("--task", "-t", type=str, default="hybrid")
parser.add_argument("--device", "-d", type=str, default="sim")
args = parser.parse_args()

save_pth = "results"


if __name__ == "__main__":
    data = JobshopData(args.job_num, args.machine_num)
    if args.task == "hybrid":
        if args.device not in ["sim", "real"]:
            raise ValueError("device must be 'sim' or 'real'!")
        solver = QCMILPSolver(args.job_num, args.machine_num, data, args.device)
        solver.solve(save_pth)
        plot_gantt_chart(data, args.job_num, args.machine_num, args.task)
    elif args.task == "classic":
        solver = ClassicSolver(args.job_num, args.machine_num)
        solver.solve(save_pth)
        plot_gantt_chart(data, args.job_num, args.machine_num, args.task)
    else:
        raise ValueError("task must be 'hybrid' or 'classic'!")
