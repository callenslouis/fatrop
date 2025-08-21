import json
import matplotlib.pyplot as plt
import numpy as np

def visualize_ocp_result(data, **kwargs):
    axs = kwargs.get('axs', None)
    if axs is None:
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    tt = np.linspace(0, data['K'] * data['dt'], data['K'])
    
    axs[0].plot(tt, [x[0] for x in data['states']], label='px')
    axs[0].plot(tt, [x[1] for x in data['states']], label='py')
    axs[0].set_ylabel('Position')
    axs[0].legend()

    axs[1].plot(tt, [x[2] for x in data['states']], label='vx')
    axs[1].plot(tt, [x[3] for x in data['states']], label='vy')
    axs[1].set_ylabel('Velocity')
    axs[1].legend()

    axs[2].plot(tt[:-1], [u[0] for u in data['inputs']], label='ux')
    axs[2].plot(tt[:-1], [u[1] for u in data['inputs']], label='uy')
    axs[2].set_ylabel('Control')
    axs[2].set_xlabel('Time')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file = 'build/unittest/implicit_ocp_result.json'
    with open(file, 'r') as f:
        data = json.load(f)

    visualize_ocp_result(data)
    
