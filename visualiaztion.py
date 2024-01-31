import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from data import JobshopData

random.seed(0)

root_pth = "results"
plot_pth = "images"

def plot_gantt_chart(data: JobshopData, job_num: int, machine_num: int, task: str) -> None:
    """
    Plot and save a Gantt chart for the given jobshop data.
    
    Args:
        data (JobshopData): The jobshop data to be used for plotting.
        job_num (int): The number of jobs in the data.
        machine_num (int): The number of machines in the data.
        task (str): The type of task, either "classic" or "hybrid".
        
    Returns:
        None
    """
    if task == "classic":
        result_pth = f"C_{job_num}_{machine_num}.json"
    else:
        result_pth = f"QC_{job_num}_{machine_num}.json"
    
    result = os.path.join(root_pth, result_pth)
    with open(result, 'r') as f:
        solver_data = json.load(f)
    instance_num = solver_data["instance_num"]
    xs = solver_data["optimal_xs"]
    ts = solver_data["optimal_ts"]
    colors = [mcolors.CSS4_COLORS[random.choice(list(mcolors.CSS4_COLORS.keys()))] for _ in range(job_num)]

    for i in range(instance_num):
        x = np.array(xs[i])
        t = np.array(ts[i])
        p_arr, _, _, _ = data.instances[i]
        fig, ax = plt.subplots(figsize=(10, 15))

        for j in range(job_num):
            m = np.nonzero(x[j])[0].item()
            ax.broken_barh([(t[j], p_arr[j, m])], (m-0.2, 0.4), facecolors=colors[j])

        ax.set_xlabel('Time')
        ax.set_ylabel('Machine')
        ax.set_title('Gantt Chart of Job Shop Scheduling')
        ax.set_yticks(range(1, machine_num+1))
        ax.set_yticklabels([f'Machine {i}' for i in range(1, machine_num+1)])
        ax.set_xlim(0, 250)
        ax.set_ylim(0.5, machine_num + 0.5)

        legend_patches = [patches.Patch(color=colors[i], label=f'Job {i+1}') for i in range(job_num)]
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2)

        plt.tight_layout()
        if not os.path.exists(plot_pth):
            os.makedirs(plot_pth)
        save_pth = os.path.join(plot_pth, f"{task}_{job_num}_{machine_num}_{i+1}.png")
        plt.savefig(save_pth)
        plt.show()


if __name__ == '__main__':
    data = JobshopData(120, 90)
    plot_gantt_chart(data, 120, 90, "hybrid")
