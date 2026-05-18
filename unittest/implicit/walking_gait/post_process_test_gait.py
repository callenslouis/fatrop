import pickle as pkl
import matplotlib.pyplot as plt
import os
import numpy as np

# FOLDER = 'stored_solutions'
FOLDER = 'unittest/implicit/walking_gait/stored_solutions'

# obtain all stats files for one ocp type
def get_stats_for_ocp_type(file_appendix):
    stats_list = []
    files = [f for f in os.listdir(FOLDER) if f.startswith('solution_gait_shortcut_reformulated_' + file_appendix)]
    for file in files:
        with open(f'{FOLDER}/{file}', 'rb') as f:
            data = pkl.load(f)
            stats_list.append(data['stats'])
    print(stats_list[0].keys())
    print()
    print(stats_list[0]['fatrop'].keys())
    print()
    return stats_list

# get averages and std for each timing metric across all runs for each ocp type
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
        if isinstance(stats_lists[key][0], dict):
            avg_stats[key] = average_stats(stats_lists[key])  # recursive averaging for nested dicts
        elif isinstance(stats_lists[key][0], (int, float)):
            avg_stats[key]['all'] = stats_lists[key]
            avg_stats[key]['mean'] = np.mean(stats_lists[key])
            avg_stats[key]['std'] = np.std(stats_lists[key])
        else:
            # print(f"Warning: Unsupported type for averaging for key '{key}'")
            pass
    
    return avg_stats

# get all stats for both ocp types
# multiple stats exist for each type
def get_all_stats():
    ocp_type_stats = get_stats_for_ocp_type('ocp_type')
    accelerated_ocp_type_stats = get_stats_for_ocp_type('accelerated_ocp_type')
    
    ocp_average_stats = average_stats(ocp_type_stats)
    accelerated_ocp_type_stats = average_stats(accelerated_ocp_type_stats)
    
    return ocp_type_stats, accelerated_ocp_type_stats, ocp_average_stats, accelerated_ocp_type_stats

def filter(avg_stats):
    filtered_stats = {'fatrop': {}}
    casadi_func_keys = ['t_wall_nlp_f', 't_wall_nlp_g', 't_wall_nlp_grad', 't_wall_nlp_grad_f', 't_wall_nlp_hess_l', 't_wall_nlp_jac_g']
    fatrop_func_keys = ['eval_cv_', 'eval_grad_', 'eval_hess_', 'eval_jac_', 'eval_obj_']
    casadi_other_keys = ['t_wall_total']
    fatrop_other_keys = ['compute_sd_time', 'time_total']
    
    for key in casadi_func_keys:
        # if key in avg_stats:
        filtered_stats[key] = avg_stats[key]
    for key in casadi_other_keys:
        if key in avg_stats:
            filtered_stats[key] = avg_stats[key]
    for key in fatrop_func_keys:
        if f'{key}time' in avg_stats['fatrop']:
            filtered_stats['fatrop'][f'{key}time'] = avg_stats['fatrop'][f'{key}time']
            # filtered_stats['fatrop'][f'{key}count'] = avg_stats['fatrop'][f'{key}count']
    for key in fatrop_other_keys:
        if key in avg_stats['fatrop']:
            filtered_stats['fatrop'][key] = avg_stats['fatrop'][key]
            
    # compute total function evaluation times
    filtered_stats['func_eval'] = { 'mean': sum(avg_stats[f'{key}']['mean'] for key in casadi_func_keys),
                                    'std': np.sqrt(sum(avg_stats[f'{key}']['std']**2 for key in casadi_func_keys)),
                                    'all': [avg_stats[f'{key}']['all'] for key in casadi_func_keys]}
    filtered_stats['fatrop']['func_eval'] = {'mean': sum(avg_stats['fatrop'][f'{key}time']['mean'] for key in fatrop_func_keys),
                                             'std': np.sqrt(sum(avg_stats['fatrop'][f'{key}time']['std']**2 for key in fatrop_func_keys)),
                                             'all': [avg_stats['fatrop'][f'{key}time']['all'] for key in fatrop_func_keys]}
    return filtered_stats

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

ocp_stats, accelerated_ocp_stats, ocp_avg, accelerated_ocp_avg = get_all_stats()
ocp_filtered, accelerated_ocp_filtered = filter(ocp_avg), filter(accelerated_ocp_avg)
print_comparison(ocp_filtered['fatrop'], accelerated_ocp_filtered['fatrop'], label1='OCP Type', label2='Accelerated OCP Type')

# timings_ocp_type, timings_accelerated_ocp_type = load_solutions()
# print_comparison(timings_ocp_type, timings_accelerated_ocp_type)

# visualize computation times
plt.figure()

# show boxplots for compute_sd_time, func_eval_time, and fatrop_time
labels = ['compute_sd_time', 'func_eval', 'time_total']
data_ocp_type = [ocp_filtered['fatrop'][label]['all'] for label in labels]
data_accelerated_ocp_type = [accelerated_ocp_filtered['fatrop'][label]['all'] for label in labels]

print(data_ocp_type)
print(len(data_ocp_type))
print([len(i) for i in data_ocp_type])

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