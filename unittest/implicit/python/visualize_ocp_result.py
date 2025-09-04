import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import pandas as pd

def visualize_holonomic_result(data, **kwargs):
    dimension = data["generator_data"]["n"]
    control_level = data["generator_data"]["control_level"]
    problem_type = data["problem type"]
    solver = data["solver"]
    problem_name = data["generator_data"]["problem_name"]

    axs = kwargs.get('axs', None)
    if axs is None:
        fig, axs = plt.subplots(control_level+1, 1)

    tt = np.linspace(0, data["generator_data"]['K'] * data["generator_data"]['dt'], data["generator_data"]['K'])

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
    plt.savefig(f"unittest/implicit/figures/ocp_result_{problem_name}_dim{dimension}_ctl{control_level}_{problem_type}_{solver}.png", dpi=300)
    plt.close()

def visualize_trucktrailer_result(data, **kwargs):
    n = data["generator_data"]["n"]
    problem_type = data["problem type"]
    solver = data["solver"]
    problem_name = data["generator_data"]["problem_name"]
    L = 1.0
    M = 0.0

    axs = kwargs.get('axs', None)
    if axs is None:
        fig, axs = plt.subplots(2, 1, height_ratios=[3, 1])

    tt = np.linspace(0, data["generator_data"]['K'] * data["generator_data"]['dt'], data["generator_data"]['K'])

    th = np.array([data['states'][k][:-2] for k in range(data["generator_data"]['K'])]).transpose()
    xy = np.array([data['states'][k][-2:] for k in range(data["generator_data"]['K'])]).transpose()

    xx = [[] for _ in range(n+1)]   # xx[veh_idx][time_idx]
    yy = [[] for _ in range(n+1)]
    th = [[th[i,k] for k in range(data["generator_data"]['K'])] for i in range(n+1)]  # th[veh_idx][time_idx]

    for k in range(data["generator_data"]['K']):
        # position of last trailer
        x = xy[0, k]
        y = xy[1, k]

        xx[n].append(x)
        yy[n].append(y)

        for i in range(n-1, -1, -1):
            xx[i].append(xx[i+1][k] + L*np.cos(th[i+1][k]) + M*np.cos(th[i][k]))
            yy[i].append(yy[i+1][k] + L*np.sin(th[i+1][k]) + M*np.sin(th[i][k]))

    # plot trajectory of each trailer
    for i in range(n+1):
        axs[0].plot(xx[i], yy[i], label=f'vehicle {i}')
        step = 5
        axs[0].quiver(xx[i][::step], yy[i][::step], np.cos(th[i][::step]), np.sin(th[i][::step]), scale=20, width=0.003, zorder=5)
    axs[0].set_aspect('equal', 'box')
    axs[0].legend()
    
    # plot control
    axs[1].plot(tt[:-1], [u[0] for u in data['inputs']], label='v')
    axs[1].plot(tt[:-1], [u[1] for u in data['inputs']], label='w')
    axs[1].legend()

    plt.suptitle(f"{problem_type} - {solver}")

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"unittest/implicit/figures/ocp_result_{problem_name}_n_{n}_{problem_type}_{solver}.png", dpi=300)
    plt.close()

def visualize_planar_robot_result(data, **kwargs):
    n = data["generator_data"]["n"]
    problem_type = data["problem type"]
    solver = data["solver"]
    problem_name = data["generator_data"]["problem_name"]

    axs = kwargs.get('axs', None)
    if axs is None:
        fig, axs = plt.subplots(2, 1, height_ratios=[3, 1])

    tt = np.linspace(0, data["generator_data"]['K'] * data["generator_data"]['dt'], data["generator_data"]['K'])

    # get positions of joints
    angles = np.array([data['states'][k][:n] for k in range(data["generator_data"]['K'])]).transpose()
    print(data["generator_data"])
    link_length = data["generator_data"]["l"]
    xx = [[] for _ in range(n+1)]   # xx[joint_idx][time_idx]
    yy = [[] for _ in range(n+1)]
    for k in range(data["generator_data"]['K']):
        x, y = 0, 0
        xx[0].append(x)
        yy[0].append(y)
        current_angle = 0
        for i in range(n):
            # current_angle = angles[i,k]
            current_angle = data['states'][k][i]
            x -= link_length * np.sin(current_angle)
            y += link_length * np.cos(current_angle)
            xx[i+1].append(x)
            yy[i+1].append(y)

    start_color = np.array([1, 0, 0])
    end_color = np.array([0, 0, 1])
    step = 15
    for k in range(0, data["generator_data"]['K'], step):
        c = k/data["generator_data"]['K']
        color = start_color * (1 - c) + end_color * c
        axs[0].plot([xx[i][k] for i in range(n+1)], [yy[i][k] for i in range(n+1)], 'o-', color=color)

    axs[0].set_aspect('equal', 'box')
    axs[0].legend()
    
    # plot control
    for j in range(n):
        axs[1].plot(tt[:-1], [u[j] for u in data['inputs']], label='u'+str(j))
    axs[1].legend()

    plt.suptitle(f"{problem_type} - {solver}")

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"unittest/implicit/figures/ocp_result_{problem_name}_n_{n}_{problem_type}_{solver}.png", dpi=300)
    plt.close()

def animate_planar_robot_result(data, **kwargs):
    n = data["generator_data"]["n"]
    problem_type = data["problem type"]
    solver = data["solver"]
    problem_name = data["generator_data"]["problem_name"]

    axs = kwargs.get('axs', None)
    if axs is None:
        fig, axs = plt.subplots(1, 1, height_ratios=[1])

    tt = np.linspace(0, data["generator_data"]['K'] * data["generator_data"]['dt'], data["generator_data"]['K'])

    # get positions of joints
    angles = np.array([data['states'][k][:n] for k in range(data["generator_data"]['K'])]).transpose()
    print(data["generator_data"])
    link_length = data["generator_data"]["l"]
    xx = [[] for _ in range(n+1)]   # xx[joint_idx][time_idx]
    yy = [[] for _ in range(n+1)]
    for k in range(data["generator_data"]['K']):
        x, y = 0, 0
        xx[0].append(x)
        yy[0].append(y)
        current_angle = 0
        for i in range(n):
            # current_angle = angles[i,k]
            current_angle = data['states'][k][i]
            x -= link_length * np.sin(current_angle)
            y += link_length * np.cos(current_angle)
            xx[i+1].append(x)
            yy[i+1].append(y)

    def update(frame):
        axs.clear()
        start_color = np.array([1, 0, 0])
        end_color = np.array([0, 0, 1])
        step = 15
        k = frame
        c = k/data["generator_data"]['K']
        color = start_color * (1 - c) + end_color * c
        plt.plot([xx[i][k] for i in range(n+1)], [yy[i][k] for i in range(n+1)], 'o-', color=color)
        axs.set_aspect('equal', 'box')
        axs.set_xlim([-(n+1)*link_length, (n+1)*link_length])
        axs.set_ylim([-(n+1)*link_length, (n+1)*link_length])
        axs.set_title(f'Time: {tt[frame]:.2f}s')

    sim_ms_per_frame = 100
    real_ms_per_frame = data["generator_data"]['dt'] * 1000
    frame_step = max(1, int(sim_ms_per_frame / real_ms_per_frame))
    ani = FuncAnimation(fig, update, frames=range(0, data["generator_data"]['K'], frame_step), interval=sim_ms_per_frame)
    ani.save(f"unittest/implicit/figures/ocp_result_{problem_name}_n_{n}_{problem_type}_{solver}_animation.gif", writer='imagemagick', fps=10)
    plt.close()

def visualize_performance(df):
    nx_vals = df['nx'].unique()
    nx_vals = np.sort(nx_vals)
    problem_name = df["problem_name"][0]
    print(f"Visualizing performance for problem: {problem_name}")

    # problem_types = df['problem type'].unique()
    problem_types = ['explicit', 'implicit', 'reformulated']

    colors = {'implicit': ['blue'],
              'explicit': ['red'],
              'reformulated': ['black']}
    
    # show solver times
    plt.figure()
    for i, problem_type in enumerate(problem_types):
        times = []
        df_pt = df[df['problem type'] == problem_type]
        for nx in nx_vals:
            df_nx = df_pt[df_pt['nx'] == nx]
            times.append((df_nx['time_solver'] / df_nx['nb_iterations']).mean())
    
        times = np.array(times)

        bar_width = 0.2
        index = np.arange(len(nx_vals)) + i*bar_width
        bb = np.zeros(len(nx_vals))
        plt.bar(index, times, bar_width, bottom=bb, label=f'{problem_type}', color=colors[problem_type][0])    
    plt.xlabel('Number of state variables (nx)')
    plt.ylabel('average solver time per iteration (s)')
    plt.title(problem_name)
    plt.xticks(index + bar_width / 2, nx_vals)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"unittest/implicit/figures/ocp_{problem_name}_performance_comparison_avg_solver_times.png", dpi=300)
    plt.close()

    # show function evaluation times
    plt.figure()
    for i, problem_type in enumerate(problem_types):
        times = []
        df_pt = df[df['problem type'] == problem_type]
        for nx in nx_vals:
            df_nx = df_pt[df_pt['nx'] == nx]
            times.append((df_nx['time_function_evaluation'] / df_nx['nb_iterations']).mean())
    
        times = np.array(times)

        bar_width = 0.2
        index = np.arange(len(nx_vals)) + i*bar_width
        bb = np.zeros(len(nx_vals))
        plt.bar(index, times, bar_width, bottom=bb, label=f'{problem_type}', color=colors[problem_type][0])    
    plt.xlabel('Number of state variables (nx)')
    plt.ylabel('average func eval time per iteration (s)')
    plt.title(problem_name)
    plt.xticks(index + bar_width / 2, nx_vals)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"unittest/implicit/figures/ocp_{problem_name}_performance_comparison_avg_func_eval_times.png", dpi=300)
    plt.close()

    # show nb of iterations
    plt.figure()
    for i, problem_type in enumerate(problem_types):
        times = []
        df_pt = df[df['problem type'] == problem_type]
        for nx in nx_vals:
            df_nx = df_pt[df_pt['nx'] == nx]
            times.append((df_nx['nb_iterations']).mean())
    
        times = np.array(times)

        bar_width = 0.2
        index = np.arange(len(nx_vals)) + i*bar_width
        bb = np.zeros(len(nx_vals))
        plt.bar(index, times, bar_width, bottom=bb, label=f'{problem_type}', color=colors[problem_type][0])    
    plt.xlabel('Number of state variables (nx)')
    plt.ylabel('nb of iterations')
    plt.title(problem_name)
    plt.xticks(index + bar_width / 2, nx_vals)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"unittest/implicit/figures/ocp_{problem_name}_performance_comparison_nb_iterations.png", dpi=300)
    plt.close()

    # show total time
    plt.figure()
    for i, problem_type in enumerate(problem_types):
        times = []
        df_pt = df[df['problem type'] == problem_type]
        for nx in nx_vals:
            df_nx = df_pt[df_pt['nx'] == nx]
            times.append((df_nx['time_total']).mean())
    
        times = np.array(times)

        bar_width = 0.2
        index = np.arange(len(nx_vals)) + i*bar_width
        bb = np.zeros(len(nx_vals))
        plt.bar(index, times, bar_width, bottom=bb, label=f'{problem_type}', color=colors[problem_type][0])    
    plt.xlabel('Number of state variables (nx)')
    plt.ylabel('total time (s)')
    plt.title(problem_name)
    plt.xticks(index + bar_width / 2, nx_vals)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"unittest/implicit/figures/ocp_{problem_name}_performance_comparison_total_time.png", dpi=300)
    plt.close()

def print_performance_table(df):
    # rows: nb iterations, t_func_avg, t_func_total, t_fatrop_avg, t_fatrop_total, t_total
    # columns: explicit, implicit, reformulated
    problem_types = ['explicit', 'implicit', 'reformulated']
    performance_data = {}

    for problem_type in problem_types:
        df_pt = df[df['problem type'] == problem_type]
        performance_data[problem_type] = [
            df_pt['nb_iterations'].mean(),
            df_pt['time_function_evaluation'].mean()/df_pt['nb_iterations'].mean(),
            df_pt['time_function_evaluation'].mean(),
            df_pt['time_solver'].mean()/df_pt['nb_iterations'].mean(),
            df_pt['time_solver'].mean(),
            df_pt['time_total'].mean()
        ]

    performance_df = pd.DataFrame(performance_data, index=[
        'nb iterations', 
        't_func_avg (s)', 
        't_func_total (s)', 
        't_fatrop_avg (s)', 
        't_fatrop_total (s)', 
        't_total (s)'
    ])

    print(performance_df.to_markdown())
    print("")

def print_holonomic_result_differences(df):
    problem_types = ['explicit', 'implicit', 'reformulated']
    results = {}
    baseline = 'implicit'

    for problem_type in problem_types:
        if problem_type == baseline:
            continue

        df_pt = df[df['problem type'] == problem_type]
        df_bl = df[df['problem type'] == baseline]

        avg_err_states = 0
        avg_err_inputs = 0
        nb_entries = 0
        
        # ensure same dimensions and control levels
        for (dim, ctl) in df_pt[['dimension', 'control level']].values:
            df_pt_dc = df_pt[(df_pt['dimension'] == dim) & (df_pt['control level'] == ctl)]
            df_bl_dc = df_bl[(df_bl['dimension'] == dim) & (df_bl['control level'] == ctl)]

            if df_pt_dc.empty or df_bl_dc.empty:
                continue

            # compare states and inputs
            states_bl = np.array(df_bl_dc['states'].tolist()).flatten()
            inputs_bl = np.array(df_bl_dc['inputs'].tolist()).flatten()

            if problem_type == 'reformulated':
                states_pt = np.array(df_pt_dc['states'].tolist()).flatten()
                nu = int(df_bl_dc['nu'].values[0])
                K = df_bl_dc['K'].values[0]
                inputs_pt = np.zeros((nu*(K-1)))

                for k in range(K-1):
                    inputs_pt[nu*k:nu*(k+1)] = np.array(df_pt_dc['inputs'].to_list())[:, k, :nu]
                inputs_pt = inputs_pt.flatten()
            else:
                states_pt = np.array(df_pt_dc['states'].tolist()).flatten()
                inputs_pt = np.array(df_pt_dc['inputs'].tolist()).flatten()

            states_diff = np.abs(states_bl - states_pt)
            inputs_diff = np.abs(inputs_bl - inputs_pt)

            avg_err_states += np.linalg.norm(states_diff)
            avg_err_inputs += np.linalg.norm(inputs_diff)
            nb_entries += 1

        if nb_entries > 0:
            results[problem_type] = {'state_diff_norm': avg_err_states/nb_entries, 'input_diff_norm': avg_err_inputs/nb_entries}
        else:
            results[problem_type] = {'state_diff_norm': -1, 'input_diff_norm': -1}

    df = pd.DataFrame.from_dict(results, orient='index', columns=['state_diff_norm', 'input_diff_norm'])
    print(df.to_markdown())
    print("")

def print_trucktrailer_result_differences(df):
    problem_types = ['explicit', 'implicit', 'reformulated']
    results = {}
    baseline = 'implicit'

    for problem_type in problem_types:
        if problem_type == baseline:
            continue

        df_pt = df[df['problem type'] == problem_type]
        df_bl = df[df['problem type'] == baseline]

        avg_err_states = 0
        avg_err_inputs = 0
        nb_entries = 0
        
        # ensure same dimensions and control levels
        for n in df_pt['n'].values:
            df_pt_dc = df_pt[df_pt['n'] == n]
            df_bl_dc = df_bl[df_bl['n'] == n]

            if df_pt_dc.empty or df_bl_dc.empty:
                continue

            # compare states and inputs
            states_bl = np.array(df_bl_dc['states'].tolist()).flatten()
            inputs_bl = np.array(df_bl_dc['inputs'].tolist()).flatten()

            if problem_type == 'reformulated':
                states_pt = np.array(df_pt_dc['states'].tolist()).flatten()
                nu = int(df_bl_dc['nu'].values[0])
                K = df_bl_dc['K'].values[0]
                inputs_pt = np.zeros((nu*(K-1)))

                for k in range(K-1):
                    inputs_pt[nu*k:nu*(k+1)] = np.array(df_pt_dc['inputs'].to_list())[:, k, :nu]
                inputs_pt = inputs_pt.flatten()
            else:
                states_pt = np.array(df_pt_dc['states'].tolist()).flatten()
                inputs_pt = np.array(df_pt_dc['inputs'].tolist()).flatten()

            states_diff = np.abs(states_bl - states_pt)
            inputs_diff = np.abs(inputs_bl - inputs_pt)

            avg_err_states += np.linalg.norm(states_diff)
            avg_err_inputs += np.linalg.norm(inputs_diff)
            nb_entries += 1

        if nb_entries > 0:
            results[problem_type] = {'state_diff_norm': avg_err_states/nb_entries, 'input_diff_norm': avg_err_inputs/nb_entries}
        else:
            results[problem_type] = {'state_diff_norm': -1, 'input_diff_norm': -1}

    df = pd.DataFrame.from_dict(results, orient='index', columns=['state_diff_norm', 'input_diff_norm'])
    print(df.to_markdown())
    print("")


if __name__ == "__main__":
    # find all files in build/unittest/ocp_results/
    dir_path = os.path.join(os.path.dirname(__file__), '../../../build/unittest/ocp_results')

    # store timing info in dataframe
    df_holonomic = pd.DataFrame(columns=['problem_name', 'problem type', 
                                         'solver', 'dimension', 
                                         'control level', 'K', 'nx', 'nu',
                                         'time_total', 
                                         'time_solver', 
                                         'time_function_evaluation', 
                                         'compute_search_dir',
                                         'nb_iterations',
                                         'states', 'inputs'])
    df_trucktrailer = pd.DataFrame(columns=['problem_name', 'problem type', 
                                            'solver', 'n',
                                            'K', 'nx', 'nu',
                                            'time_total', 
                                            'time_solver', 
                                            'time_function_evaluation', 
                                            'compute_search_dir',
                                            'nb_iterations',
                                            'states', 'inputs'])
    
    df_planarrobot = pd.DataFrame(columns=['problem_name', 'problem type',
                                            'solver', 'n',
                                            'K', 'nx', 'nu', 'l', 
                                            'time_total', 
                                            'time_solver', 
                                            'time_function_evaluation', 
                                            'compute_search_dir',
                                            'nb_iterations',
                                            'states', 'inputs'])

    for file_name in os.listdir(dir_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(dir_path, file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)

            if data["generator_data"]["problem_name"] == "holonomic":
                df_holonomic.loc[len(df_holonomic)] = {
                    'problem_name': data["generator_data"]["problem_name"],
                    'problem type': data['problem type'],
                    'solver': data['solver'],
                    'dimension': data["generator_data"]["n"],
                    'control level': data["generator_data"]["control_level"],
                    'K': data["generator_data"]['K'],
                    'nx': data["generator_data"]['nx'], 
                    'nu': data["generator_data"]['nu'],
                    'time_total': data["metadata"]["timing_statistics"]['total'],
                    'time_solver': data["metadata"]["timing_statistics"]['fatrop'],
                    'time_function_evaluation': data["metadata"]["timing_statistics"]['function evaluation'],
                    'compute_search_dir': data["metadata"]["timing_statistics"]['compute search dir'],
                    'nb_iterations': data["metadata"]['iterations'],
                    'states': data['states'],
                    'inputs': data['inputs']
                }
                
                # visualize_holonomic_result(data)
            
            elif data["generator_data"]["problem_name"] == "truck_trailer":
                df_trucktrailer.loc[len(df_trucktrailer)] = {
                    'problem_name': data["generator_data"]["problem_name"],
                    'problem type': data['problem type'],
                    'solver': data['solver'],
                    'n': data["generator_data"]["n"],
                    'K': data["generator_data"]['K'], 
                    'nx': data["generator_data"]['nx'], 
                    'nu': data["generator_data"]['nu'],
                    'time_total': data["metadata"]["timing_statistics"]['total'],
                    'time_solver': data["metadata"]["timing_statistics"]['fatrop'],
                    'time_function_evaluation': data["metadata"]["timing_statistics"]['function evaluation'],
                    'compute_search_dir': data["metadata"]["timing_statistics"]['compute search dir'],
                    'nb_iterations': data["metadata"]['iterations'],
                    'states': data['states'],
                    'inputs': data['inputs']
                }
            
                # visualize_trucktrailer_result(data)

            elif data["generator_data"]["problem_name"] == "planar_robot":
                df_planarrobot.loc[len(df_planarrobot)] = {
                    'problem_name': data["generator_data"]["problem_name"],
                    'problem type': data['problem type'],
                    'solver': data['solver'],
                    'n': data["generator_data"]["n"],
                    'K': data["generator_data"]['K'], 
                    'nx': data["generator_data"]['nx'], 
                    'nu': data["generator_data"]['nu'],
                    'l': data["generator_data"]['l'],
                    'time_total': data["metadata"]["timing_statistics"]['total'],
                    'time_solver': data["metadata"]["timing_statistics"]['fatrop'],
                    'time_function_evaluation': data["metadata"]["timing_statistics"]['function evaluation'],
                    'compute_search_dir': data["metadata"]["timing_statistics"]['compute search dir'],
                    'nb_iterations': data["metadata"]['iterations'],
                    'states': data['states'],
                    'inputs': data['inputs']
                }

                # visualize_planar_robot_result(data)
                # animate_planar_robot_result(data)
            
            else:
                print(f"Unknown problem name: {data["generator_data"]['problem_name']}")

    if (not df_holonomic.empty):
        visualize_performance(df_holonomic)
        print_performance_table(df_holonomic)
        print_holonomic_result_differences(df_holonomic)

    if (not df_trucktrailer.empty):
        visualize_performance(df_trucktrailer)
        print_performance_table(df_trucktrailer)
        print_trucktrailer_result_differences(df_trucktrailer)

    if (not df_planarrobot.empty):
        visualize_performance(df_planarrobot)
        print_performance_table(df_planarrobot)
        # print_trucktrailer_result_differences(df_planarrobot)
