import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def visualize_ocp_result(data, **kwargs):
    dimension = data["ocp problem"]["number of dimensions"]
    control_level = data["ocp problem"]["control level"]
    problem_type = data["problem type"]
    solver = data["solver"]

    axs = kwargs.get('axs', None)
    if axs is None:
        fig, axs = plt.subplots(control_level+1, 1)

    tt = np.linspace(0, data['K'] * data["ocp problem"]['dt'], data['K'])

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
    plt.savefig(f"unittest/implicit/figures/ocp_result_dim{dimension}_ctl{control_level}_{problem_type}_{solver}.png", dpi=300)
    plt.close()

def visualize_performance(df):
    plt.figure()
    nx_vals = df['nx'].unique()
    nx_vals = np.sort(nx_vals)

    # problem_types = df['problem type'].unique()
    problem_types = ['explicit', 'implicit', 'reformulated']

    colors = {'implicit': ['lightblue', 'darkblue'],
              'explicit': ['red', 'darkred'],
              'reformulated': ['gray', 'black']}
    
    # SHOW TIMINGS
    keys_to_show = ['time_solver', 'time_function_evaluation']
    for i, problem_type in enumerate(problem_types):
        times = [[] for _ in range(len(keys_to_show))]
        df_pt = df[df['problem type'] == problem_type]
        for nx in nx_vals:
            df_nx = df_pt[df_pt['nx'] == nx]
            for j in range(len(keys_to_show)):
                times[j].append(df_nx[keys_to_show[j]].mean())
        
        for j in range(len(times)):
            times[j] = np.array(times[j])

        bar_width = 0.2
        index = np.arange(len(nx_vals)) + i*bar_width
        bb = np.zeros(len(nx_vals))
        for j in range(len(times)):
            plt.bar(index, times[j], bar_width, bottom=bb, label=f'{problem_type} - {keys_to_show[j]}', color=colors[problem_type][j % len(colors[problem_type])])    
            bb += np.array(times[j])
    plt.xlabel('Number of state variables (nx)')
    plt.ylabel('Time (s)')
    plt.title('Performance comparison')
    plt.xticks(index + bar_width / 2, nx_vals)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"unittest/implicit/figures/ocp_performance_comparison_times.png", dpi=300)


    # SHOW ITERATION BRAKWDOWN
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    keys_to_show = ['nb_iterations', 'time_solver', 'time_function_evaluation']
    for i, problem_type in enumerate(problem_types):
        times = [[] for _ in range(len(keys_to_show))]
        df_pt = df[df['problem type'] == problem_type]
        for nx_idx, nx in enumerate(nx_vals):
            df_nx = df_pt[df_pt['nx'] == nx]
            for j in range(len(keys_to_show)):
                if j == 0:
                    times[j].append(df_nx[keys_to_show[j]].mean())
                else:
                    times[j].append(df_nx[keys_to_show[j]].mean() / times[0][nx_idx])
        
        for j in range(len(times)):
            times[j] = np.array(times[j])

        bar_width = 0.2
        index = np.arange(len(nx_vals)) + i*bar_width
        bb = np.zeros(len(nx_vals))
        for j in range(len(times)):
            if j == 0:
                # show this on a separate y-axis to preserve scaling
                ax2.bar(index, times[j], bar_width/len(times), bottom=0*bb, label=f'{problem_type} - {keys_to_show[j]}', color='lightgray')
            if j > 0:
                ax1.bar(index + j*bar_width/len(times), times[j], bar_width/len(times), bottom=0*bb, 
                        label=f'{problem_type} - {keys_to_show[j]}', 
                        color=colors[problem_type][j % len(colors[problem_type])])    
            bb += np.array(times[j])
    ax1.set_xlabel('Number of state variables (nx)')
    ax1.set_ylabel('Time (s)')
    ax2.set_ylabel('Number of iterations')
    plt.title('Performance comparison')
    ax1.set_xticks(index + bar_width / 2, nx_vals)
    
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # ax1.legend(h1+h2, l1+l2)
    
    fig.tight_layout()
    plt.savefig(f"unittest/implicit/figures/ocp_performance_comparison_iterations.png", dpi=300)

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

def print_result_differences(df):
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

            if np.linalg.norm(inputs_diff) > 1:
                print(f"Significant input difference for {problem_type} with dim {dim} and ctl {ctl}")
                # for i in range(len(inputs_diff)):
                #     print(f"[{i}] err: {inputs_diff[i]:.3f}, reformulated: {inputs_pt[i]:.6f}, implicit: {inputs_bl[i]:.6f}")
                for i in range(K-1):
                    b = np.array(df_bl_dc['inputs'].to_list())
                    p = np.array(df_pt_dc['inputs'].to_list())
                    e = np.linalg.norm(b[:,i,:] - p[:,i,:nu])
                    print(f"e: {e}\t-\tb: {b[:,i,:]}, p: {p[:,i,:nu]}")


            avg_err_states += np.linalg.norm(states_diff)
            avg_err_inputs += np.linalg.norm(inputs_diff)
            nb_entries += 1

        results[problem_type] = {'state_diff_norm': avg_err_states/nb_entries, 'input_diff_norm': avg_err_inputs/nb_entries}

    df = pd.DataFrame.from_dict(results, orient='index', columns=['state_diff_norm', 'input_diff_norm'])
    print(df.to_markdown())


if __name__ == "__main__":
    # find all files in build/unittest/ocp_results/
    dir_path = os.path.join(os.path.dirname(__file__), '../../build/unittest/ocp_results')

    # store timing info in dataframe
    df_holonomic = pd.DataFrame(columns=['problem type', 'solver', 'dimension', 
                                         'control level', 'K', 'nx', 'nu',
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

            if data["ocp problem"]["name"] == "holonomic":
                df_holonomic.loc[len(df_holonomic)] = {
                    'problem type': data['problem type'],
                    'solver': data['solver'],
                    'dimension': data["ocp problem"]["number of dimensions"],
                    'control level': data["ocp problem"]["control level"],
                    'K': data['K'], 'nx': data['nx'], 'nu': data['nu'],
                    'time_total': data["metadata"]["timing_statistics"]['total'],
                    'time_solver': data["metadata"]["timing_statistics"]['fatrop'],
                    'time_function_evaluation': data["metadata"]["timing_statistics"]['function evaluation'],
                    'compute_search_dir': data["metadata"]["timing_statistics"]['compute search dir'],
                    'nb_iterations': data["metadata"]['iterations'],
                    'states': data['states'],
                    'inputs': data['inputs']
                }
            
            # visualize_ocp_result(data)

    visualize_performance(df_holonomic)
    print_performance_table(df_holonomic)
    print_result_differences(df_holonomic)
