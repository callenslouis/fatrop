import json
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_ocp_result(data, **kwargs):
    dimension = data["number of dimensions"]
    control_level = data["control level"]
    problem_type = data["problem type"]
    solver = data["solver"]

    axs = kwargs.get('axs', None)
    if axs is None:
        fig, axs = plt.subplots(control_level+1, 1)

    tt = np.linspace(0, data['K'] * data['dt'], data['K'])

    state_names = ['pos', 'vel', 'acc', 'jerk', 'snap']
    for i in range(control_level):
        for j in range(dimension):
            axs[i].plot(tt, [x[i*dimension + j] for x in data['states']], label=f'{state_names[i]} ({j+1})')
        axs[i].set_xlim([tt[0], tt[-1]])
        axs[i].set_ylabel(state_names[i])
        axs[i].legend()
    
    # plot control
    for j in range(dimension):
        axs[control_level].plot(tt[:-1], [u[j] for u in data['inputs']], label=f'u ({j+1})')
        axs[control_level].set_xlim([tt[0], tt[-1]])
        axs[control_level].set_ylabel('Control')
        axs[control_level].set_xlabel('Time')
        axs[control_level].legend()

    plt.suptitle(f"{problem_type} - {solver}")

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"unittest/implicit/figures/ocp_result_{problem_type}_{solver}_dim{dimension}_ctl{control_level}.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    # find all files in build/unittest/ocp_results/
    dir_path = os.path.join(os.path.dirname(__file__), '../../build/unittest/ocp_results')

    for file_name in os.listdir(dir_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(dir_path, file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)
            visualize_ocp_result(data)

    
