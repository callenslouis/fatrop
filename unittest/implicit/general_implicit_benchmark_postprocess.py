import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    df = pd.read_csv('build_docker/random_benchmark_results.csv')

    data = {
        'K': np.array(df['K'].values),
        'nx': np.array(df['nx'].values),
        'r': np.array(df['r'].values),
        'nu': np.array(df['nu'].values),
        'ng': np.array(df['ng'].values),
        'ng_ineq': np.array(df['ng_ineq'].values),
        't_expl': np.array(df['t_expl'].values),
        't_impl': np.array(df['t_impl'].values),
        't_impl_pre': np.array(df['t_impl_pre'].values),
        't_impl_solve': np.array(df['t_impl_solve'].values),
        't_impl_post': np.array(df['t_impl_post'].values),
        't_reform': np.array(df['t_reform'].values),
    }

    return data

def visualize_scaling(data):
    metrics = ['K', 'nx', 'r', 'nu', 'ng', 'ng_ineq']
    # metrics = ['nx']

    # for each metric, plot the scaling of the times
    for metric in metrics:
        plt.figure()
        unique_metric = np.unique(data[metric])
        unique_sorted_metric = np.sort(unique_metric)

        # for each method, compute mean and std values for every metric value
        expl_means = []
        expl_stds = []
        impl_means = []
        impl_stds = []
        reform_means = []
        reform_stds = []
        for val in unique_sorted_metric:
            mask = data[metric] == val
            expl_means.append(np.mean(data['t_expl'][mask]))
            expl_stds.append(np.std(data['t_expl'][mask]))
            impl_means.append(np.mean(data['t_impl'][mask]))
            impl_stds.append(np.std(data['t_impl'][mask]))
            reform_means.append(np.mean(data['t_reform'][mask]))
            reform_stds.append(np.std(data['t_reform'][mask]))

        plt.plot(unique_sorted_metric, expl_means, label='Explicit')
        plt.fill_between(unique_sorted_metric, np.array(expl_means) - np.array(expl_stds), np.array(expl_means) + np.array(expl_stds), alpha=0.2)
        plt.plot(unique_sorted_metric, impl_means, label='Implicit')
        plt.fill_between(unique_sorted_metric, np.array(impl_means) - np.array(impl_stds), np.array(impl_means) + np.array(impl_stds), alpha=0.2)
        plt.plot(unique_sorted_metric, reform_means, label='Reformulation')
        plt.fill_between(unique_sorted_metric, np.array(reform_means) - np.array(reform_stds), np.array(reform_means) + np.array(reform_stds), alpha=0.2)
        plt.xlabel(metric)
        plt.ylabel('Time (s)')
        plt.title(f'Scaling of Times with {metric}')
        plt.legend()
        plt.grid()
        # plt.xscale('log')
        # plt.yscale('log')
        plt.show()

visualize_scaling(get_data())