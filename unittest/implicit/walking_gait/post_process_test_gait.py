import pickle as pkl
import matplotlib.pyplot as plt
import os
import numpy as np

# FOLDER = 'stored_solutions'
FOLDER = 'unittest/implicit/walking_gait/stored_solutions'

def average_stats(stats_list):
    stats_lists = {}
    for stats in stats_list:
        for key, value in stats.items():
            if key not in stats_lists:
                stats_lists[key] = []
            stats_lists[key].append(value)
    
    # Compute average
    avg_stats = {}
    for key in stats_lists:
        avg_stats[key] = {}
        avg_stats[key]['all'] = stats_lists[key]
        avg_stats[key]['mean'] = np.mean(stats_lists[key])
        avg_stats[key]['std'] = np.std(stats_lists[key])
    
    return avg_stats

def extract_relevant_timings(stats):
    relevant_keys = [
        'compute_sd_time', # time to compute search direction
        'duinf_time', # time to compute duinf
        'initialization_time', # time for initialization
    ]
    func_eval_keys = ['eval_cv_time', 'eval_grad_time', 'eval_hess_time', 'eval_jac_time', 'eval_obj_time']
    for k in func_eval_keys:
        relevant_keys.append(k)
    
    timings = {key: stats['fatrop'][key] for key in relevant_keys if key in stats['fatrop']}
    timings['func_eval_time'] = sum(timings[k] for k in func_eval_keys if k in timings)
    timings['fatrop_time'] = stats['fatrop']['time_total']
    
    return {
        'compute_sd_time': timings.get('compute_sd_time', None),
        'fatrop_time': timings.get('fatrop_time', None),
        'func_eval_time': timings.get('func_eval_time', None)
    }

def load_solutions():
    ocp_type_data = []
    files = [f for f in os.listdir(FOLDER) if f.startswith('solution_gait_shortcut_reformulated_ocp_type')]
    for file in files:
        with open(f'{FOLDER}/{file}', 'rb') as f:
            data = pkl.load(f)
            ocp_type_data.append(extract_relevant_timings(data['stats']))
    ocp_data = average_stats(ocp_type_data)

    accelerated_ocp_type_data = []
    files = [f for f in os.listdir(FOLDER) if f.startswith('solution_gait_shortcut_reformulated_accelerated_ocp_type')]
    for file in files:
        with open(f'{FOLDER}/{file}', 'rb') as f:
            data = pkl.load(f)
            accelerated_ocp_type_data.append(extract_relevant_timings(data['stats']))
    accelerated_ocp_data = average_stats(accelerated_ocp_type_data)

    return ocp_data, accelerated_ocp_data

def compute_relative_difference(time1, time2):
    if time1 is None or time2 is None:
        return None
    if time1 == 0:
        return float('inf')  # avoid division by zero, interpret as infinite improvement
    return (time1 - time2) / time1 * 100

def print_comparison(timings1, timings2, label1='OCP Type', label2='Accelerated OCP Type'):
    print(f"\n{'Timing':20} | {label1:20} | {label2:21} | {'Relative Difference (%)':20}")
    print("-" * 100)
    for key in timings1.keys():
        time1 = timings1[key]
        time2 = timings2.get(key, None)
        rel_diff = compute_relative_difference(time2['mean'], time1['mean'])
        rel_diff_str = f"{rel_diff:.2f}%" if rel_diff is not None else "N/A"
        print(f"{key:<20} | {time1['mean']:11.4f} ({time1['std']:5.4f}) | {time2['mean']:12.4f} ({time2['std']:5.4f}) | {rel_diff_str:20}")
    print()

timings_ocp_type, timings_accelerated_ocp_type = load_solutions()
print_comparison(timings_ocp_type, timings_accelerated_ocp_type)

# visualize computation times
plt.figure()

# show boxplots for compute_sd_time, func_eval_time, and fatrop_time
labels = timings_ocp_type.keys()
data_ocp_type = [timings_ocp_type[label]['all'] for label in labels]
data_accelerated_ocp_type = [timings_accelerated_ocp_type[label]['all'] for label in labels]

width = 0.6
positions_ocp = np.array(range(len(labels)))*2.0 - width/2
positions_accelerated_ocp = positions_ocp + width

plt.boxplot(data_ocp_type, 
            positions=positions_ocp, 
            widths=width, 
            patch_artist=True, 
            boxprops=dict(facecolor='lightblue'), 
            medianprops=dict(color='blue'))
plt.boxplot(data_accelerated_ocp_type, 
            positions=positions_accelerated_ocp, 
            widths=width, 
            patch_artist=True, 
            boxprops=dict(facecolor='lightgreen'), 
            medianprops=dict(color='green'))

# show true samples for each method as scatter points
for i, label in enumerate(labels):
    y_ocp = data_ocp_type[i]
    y_accelerated_ocp = data_accelerated_ocp_type[i]
    x_ocp = np.random.normal(positions_ocp[i], width/10, size=len(y_ocp))
    x_accelerated_ocp = np.random.normal(positions_accelerated_ocp[i], width/10, size=len(y_accelerated_ocp))
    plt.scatter(x_ocp, y_ocp, color='blue', alpha=0.6, label='OCP Type' if i == 0 else "", zorder=10)
    plt.scatter(x_accelerated_ocp, y_accelerated_ocp, color='green', alpha=0.6, label='Accelerated OCP Type' if i == 0 else "", zorder=10)
    

plt.xticks(np.array(range(len(labels)))*2.0, labels)
curr_ylim = plt.ylim()
plt.ylim(0, curr_ylim[1])
plt.ylabel('Time (s)')
plt.title('Computation Time Comparison')
plt.legend(['OCP Type', 'Accelerated OCP Type'])
plt.grid(axis='y')
plt.show()