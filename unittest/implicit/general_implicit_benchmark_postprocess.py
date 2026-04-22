import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random as rnd
import json
from flop_counter import FlopCounter
settings = json.load(open('unittest/implicit/post_process_settings.json', 'r'))

def latexify():
    params = {#'backend': 'ps',
              'axes.labelsize': 10,
              'axes.titlesize': 15,
              'legend.fontsize': 10,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'text.usetex': True,
              'font.family': 'serif',
              'figure.figsize': [7,5],
              'text.latex.preamble': r'\usepackage{bm}',
              }
 
    plt.rcParams.update(params)

if settings.get('latexify', True):
    latexify()

def translate_label(label):
    if label == 'K':
        return 'K'
    elif label == 'nx':
        return r'$n_x$'
    elif label == 'nu':
        return r'$n_u$'
    elif label == 'ng':
        return r'$n_g$'
    elif label == 'r':
        return r'$r$'
    elif label == 'ng_ineq':
        return r'$n_{g,ineq}$'
    else:
        raise ValueError(f'Unknown label: {label}')

def get_data():
    # file = 'build_docker/random_benchmark_results_extended.csv'
    # file = 'build_docker/random_benchmark_results.csv'
    # file = 'build_docker/random_benchmark_results_benelux.csv'
    file = 'build_docker/random_benchmark_results_huge.csv'
    df = pd.read_csv(file)

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
        # 't_accel': np.array(df['t_accel'].values),
        'lu_impl': np.array(df['lu_impl'].values),
        'lu_reform': np.array(df['lu_reform'].values),
        # 'lu_accel': np.array(df['lu_accel'].values),
        'impl_decomp': np.array(df['impl_decomp'].values),
    }

    return data

def add_flops_to_data(data):
    fc = FlopCounter()
    impl_flop, expl_flop, reform_flop = fc.get_all_flops(data)

    data['impl_flop'] = impl_flop
    data['expl_flop'] = expl_flop
    data['reform_flop'] = reform_flop

    return data

def visualize_preprocessing_scaling(data):
    flops = get_preprocessing_flop_count(data)

    metrics = ['K', 'nx', 'r', 'nu', 'ng']

    for metric in metrics:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        unique_metric = np.unique(data[metric])
        unique_sorted_metric = np.sort(unique_metric)

        impl_pre_means = []
        impl_pre_stds = []
        for val in unique_sorted_metric:
            mask = data[metric] == val
            impl_pre_means.append(np.mean(data['t_impl_pre'][mask]))
            impl_pre_stds.append(np.std(data['t_impl_pre'][mask]))

        axs[0].plot(unique_sorted_metric, impl_pre_means, label=f'Preprocessing time vs {metric}')
        axs[0].fill_between(unique_sorted_metric, np.array(impl_pre_means) - np.array(impl_pre_stds), np.array(impl_pre_means) + np.array(impl_pre_stds), alpha=0.2)
        axs[0].set_xlabel(metric)
        axs[0].set_ylabel('Preprocessing Time (s)')
        axs[0].set_title(f'Preprocessing Time Scaling with {metric}')
        axs[0].legend()

        flops_means = []
        flops_stds = []
        for val in unique_sorted_metric:
            mask = data[metric] == val
            flops_means.append(np.mean(flops[mask]))
            flops_stds.append(np.std(flops[mask]))

        axs[1].plot(unique_sorted_metric, flops_means, label=f'Preprocessing FLOP count vs {metric}')
        axs[1].set_xlabel(metric)
        axs[1].set_ylabel('Preprocessing Time (s) / FLOP Count')
        axs[1].set_title(f'Preprocessing Time and FLOP Count Scaling with {metric}')
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(f'unittest/implicit/figures_benelux/preprocessing_scaling_{metric}.png', dpi=300)

def visualize_scaling(data, **kwargs):
    metrics = ['K', 'nx', 'nu', 'ng', 'ng_ineq']
    # metrics = ['nx']

    color_explicit = 'r'
    color_implicit = 'b'
    color_reformulated = 'g'
    color_accelerated = 'yellowgreen'

    impl_flop = data['impl_flop']
    reform_flop = data['reform_flop']
    expl_flop = data['expl_flop']

    # for each metric, plot the scaling of the times
    if kwargs.get('single_figure', False):
        fig, axs = plt.subplots(2, 3)
    for i, metric in enumerate(metrics):
        if not kwargs.get('single_figure', False):
            plt.figure(figsize=(6,3.2))
            ax = plt.gca()
        else:
            # ax = axs[i//2, i%2]
            ax = axs[i//3, i%3]
        unique_metric = np.unique(data[metric])
        unique_sorted_metric = np.sort(unique_metric)

        # for each method, compute mean and std values for every metric value
        expl_means = []; impl_means = []; reform_means = []; accel_means = []
        expl_stds = []; impl_stds = []; reform_stds = []; accel_stds = []
        impl_pre_means = []; impl_pre_stds = []
        impl_solve_means = []; impl_solve_stds = []
        impl_post_means = []; impl_post_stds = []
        
        impl_flop_means = []; reform_flop_means = []; expl_flop_means = []; accel_flop_means = []

        for val in unique_sorted_metric:
            mask = data[metric] == val
            expl_means.append(np.mean(data['t_expl'][mask]))
            expl_stds.append(np.std(data['t_expl'][mask]))
            impl_means.append(np.mean(data['t_impl'][mask]))
            impl_stds.append(np.std(data['t_impl'][mask]))
            reform_means.append(np.mean(data['t_reform'][mask]))
            reform_stds.append(np.std(data['t_reform'][mask]))
            if 't_accel' in data:
                accel_means.append(np.mean(data['t_accel'][mask]))
                accel_stds.append(np.std(data['t_accel'][mask]))
            else:
                accel_means.append(0)
                accel_stds.append(0)

            impl_pre_means.append(np.mean(data['t_impl_pre'][mask]))
            impl_pre_stds.append(np.std(data['t_impl_pre'][mask]))
            impl_solve_means.append(np.mean(data['t_impl_solve'][mask]))
            impl_solve_stds.append(np.std(data['t_impl_solve'][mask]))
            impl_post_means.append(np.mean(data['t_impl_post'][mask]))
            impl_post_stds.append(np.std(data['t_impl_post'][mask]))

            if kwargs.get('show_flops', True):
                # compute relative flop-count
                mask = data[metric] == val
                impl_flop_means.append(np.mean(impl_flop[mask]))
                reform_flop_means.append(np.mean(reform_flop[mask]))
                expl_flop_means.append(np.mean(expl_flop[mask]))
                accel_flop_means.append(0)

        if kwargs.get('relative', False):
            expl_means = (np.array(expl_means) - np.array(reform_means)) / np.array(reform_means)
            impl_means = (np.array(impl_means) - np.array(reform_means)) / np.array(reform_means)
            accel_means = (np.array(accel_means) - np.array(reform_means)) / np.array(reform_means)
            if kwargs.get('show_flops', True):
                relative_flop = (np.array(impl_flop_means) - np.array(reform_flop_means)) / np.array(reform_flop_means)
                relative_flop_expl = (np.array(expl_flop_means) - np.array(reform_flop_means)) / np.array(reform_flop_means)
                relative_flop_accel = np.ones_like(relative_flop)
                ax.plot(unique_sorted_metric, relative_flop, '-', alpha=0.5, label='FLOP estimate', color=color_implicit)
                ax.plot(unique_sorted_metric, relative_flop_expl, '-', alpha=0.5, label='FLOP estimate', color=color_explicit)

            ax.plot(unique_sorted_metric, expl_means, label='Explicit (relative)', color=color_explicit)
            ax.plot(unique_sorted_metric, impl_means, label='Implicit (relative)', color=color_implicit)
            ax.plot(unique_sorted_metric, accel_means, label='Accelerated (relative)', color=color_accelerated)
            ax.axhline(0, color='k', linestyle='-')
            ax.set_ylabel('Relative\ndifference')
        else:    
            if kwargs.get('show_flops', True):
                ax.plot(unique_sorted_metric, impl_flop_means, '-', alpha=0.5, label='FLOP estimate', color=color_implicit)
                ax.plot(unique_sorted_metric, expl_flop_means, '-', alpha=0.5, label='FLOP estimate', color=color_explicit)
                ax.plot(unique_sorted_metric, reform_flop_means, '-', alpha=0.5, label='FLOP estimate', color=color_reformulated)

            ax.plot(unique_sorted_metric, expl_means, label='Explicit', color=color_explicit)
            if kwargs.get("show_std", False):
                ax.fill_between(unique_sorted_metric, np.array(expl_means) - np.array(expl_stds), np.array(expl_means) + np.array(expl_stds), alpha=0.2, color=color_explicit)
            ax.plot(unique_sorted_metric, impl_means, label='Implicit', color=color_implicit)
            if kwargs.get("show_std", False):
                ax.fill_between(unique_sorted_metric, np.array(impl_means) - np.array(impl_stds), np.array(impl_means) + np.array(impl_stds), alpha=0.2, color=color_implicit)
            ax.plot(unique_sorted_metric, impl_pre_means, ':', label='Implicit pre', color='k')
            ax.plot(unique_sorted_metric, np.sum([impl_pre_means, impl_solve_means], axis=0), '--', label='Implicit solve', color='k')
            # ax.plot(unique_sorted_metric, np.sum([impl_pre_means, impl_solve_means, impl_post_means], axis=0), ':', label='Implicit post')
            ax.plot(unique_sorted_metric, reform_means, label='Reformulation', color=color_reformulated)
            if kwargs.get("show_std", False):
                ax.fill_between(unique_sorted_metric, np.array(reform_means) - np.array(reform_stds), np.array(reform_means) + np.array(reform_stds), alpha=0.2, color=color_reformulated)
            ax.plot(unique_sorted_metric, accel_means, label='Accelerated', color=color_accelerated)
            if kwargs.get("show_std", False):
                ax.fill_between(unique_sorted_metric, np.array(accel_means) - np.array(accel_stds), np.array(accel_means) + np.array(accel_stds), alpha=0.2, color=color_accelerated)
            ax.set_ylabel('Time (ns)')

        ax.set_xlabel(translate_label(metric))
        # plt.title(f'Scaling of Times with {metric}')
        ax.grid()
        plt.tight_layout()
        if not kwargs.get('single_figure', False):
            ax.legend()
            plt.savefig(f'unittest/implicit/figures_benelux/scaling_{metric}{"_relative" if kwargs.get("relative", False) else ""}{"_flops" if kwargs.get("show_flops") else ""}.png', dpi=300)
    
    if kwargs.get('single_figure', False):
        # remove the last axis (bottom right) and put a legend there
        axs[1, 2].axis('off')
        handles, labels = [], []
        for ax in axs.flatten():
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        unique_labels = []
        unique_handles = []
        for h, l in zip(handles, labels):
            if l not in unique_labels:
                unique_labels.append(l)
                unique_handles.append(h)
        axs[1, 2].legend(unique_handles, unique_labels, loc='center')
        
        plt.savefig(f'unittest/implicit/figures_benelux/scaling_single_figure{"_relative" if kwargs.get("relative", False) else ""}{"_flops" if kwargs.get("show_flops") else ""}.png', dpi=300)

def visualize_scaling_2d(data, **kwargs):
    x = kwargs.get('x', 'nx')
    y = kwargs.get('y', 'ng')
    x_unique = np.sort(np.unique(data[x]))
    y_unique = np.sort(np.unique(data[y]))

    Z = np.full((len(y_unique), len(x_unique)), np.nan)

    method = kwargs.get('method', 'impl')

    if kwargs.get('show_flop', False):
        impl_flop = data['impl_flop']
        reform_flop = data['reform_flop']
        expl_flop = data['expl_flop']

        data_x = impl_flop
        data_y = reform_flop
    else:
        data_x = data[f't_{method}']
        data_y = data['t_reform']

    for i, y_val in enumerate(y_unique):
        for j, x_val in enumerate(x_unique):
            mask = (data[x] == x_val) & (data[y] == y_val)
            if mask.any():
                mean_method = np.mean(data_x[mask])
                mean_reform = np.mean(data_y[mask])
                if kwargs.get('relative', True):
                    Z[i, j] = (mean_method - mean_reform) / mean_reform
                else:
                    Z[i, j] = (mean_method - mean_reform)

    fig = plt.figure(figsize=(4,3))
    ax = plt.gca()

    
    level_jump = 0.1
    max_val = kwargs.get("max_val", None)
    if max_val is None:
        max_val = max(np.nanmax(Z), -np.nanmin(Z))
        max_val = min(1.5, max_val)
    cmap = plt.cm.seismic
    levels = np.arange(level_jump*(-max_val//level_jump), level_jump*(max_val//level_jump+2), level_jump)
    norm = mcolors.BoundaryNorm(levels, ncolors=256)
    cmap.set_bad(color='lightgrey')
    mesh = ax.pcolormesh(x_unique, y_unique, Z, cmap=cmap, norm=norm)
    cbar = fig.colorbar(mesh, ax=ax, label='Relative difference', )

    
    # norm = mcolors.TwoSlopeNorm(vmin=min(-0.01, np.nanmin(Z)), vcenter=0, vmax=max(np.nanmax(Z),0.01))
    # mesh = ax.pcolormesh(x_unique, y_unique, Z, cmap='bwr', norm=norm)
    # cbar = fig.colorbar(mesh, ax=ax, label='Relative difference')
    # ticks = np.linspace(norm.vmin, norm.vmax, 7)  # or choose number you like
    # cbar.set_ticks(ticks)
    ax.set_xlabel(translate_label(x))
    ax.set_ylabel(translate_label(y))
    # ax.set_title('Scaling of Implicit Time with nx and ng')
    # plt.show()
    plt.tight_layout()
    plt.savefig(f'unittest/implicit/figures_benelux/scaling_2d_{method}_{x}_{y}{"_relative" if kwargs.get("relative", False) else ""}{"_flops" if kwargs.get("show_flop") else ""}.png', dpi=300)
    

def visualize_lu_scaling(data, **kwargs):
    metrics = ['nx', 'nu', 'ng']
    method = kwargs.get('method', 'impl')

    for metric in metrics:
        plt.figure()
        unique_metric = np.unique(data[metric])
        unique_sorted_metric = np.sort(unique_metric)

        if kwargs.get('relative', False):
            lu_method_means = []
            lu_reform_means = []

            for val in unique_sorted_metric:
                mask = data[metric] == val
                lu_method_means.append(np.mean(data[f'lu_{method}'][mask]) / np.mean(data['lu_reform'][mask]))
                lu_reform_means.append(1.0)

            plt.plot(unique_sorted_metric, lu_method_means, label=f'{method} / Reformulation')
            plt.ylabel('Relative LU solve time')
        else:
            lu_method_means = []
            lu_method_stds = []
            lu_reform_means = []
            lu_reform_stds = []

            for val in unique_sorted_metric:
                mask = data[metric] == val
                lu_method_means.append(np.mean(data[f'lu_{method}'][mask]))
                lu_method_stds.append(np.std(data[f'lu_{method}'][mask]))
                lu_reform_means.append(np.mean(data['lu_reform'][mask]))
                lu_reform_stds.append(np.std(data['lu_reform'][mask]))

            plt.plot(unique_sorted_metric, lu_method_means, label=f'{method}')
            plt.plot(unique_sorted_metric, lu_reform_means, label='Reformulation')
            plt.fill_between(unique_sorted_metric, np.array(lu_method_means) - np.array(lu_method_stds), np.array(lu_method_means) + np.array(lu_method_stds), alpha=0.2)
            plt.fill_between(unique_sorted_metric, np.array(lu_reform_means) - np.array(lu_reform_stds), np.array(lu_reform_means) + np.array(lu_reform_stds), alpha=0.2)
            plt.ylabel('LU Solve Time (s)')
        
        plt.xlabel(metric)
        plt.title(f'Scaling of LU Solve Time with {metric}')
        plt.legend()
        plt.grid()
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()

def create_2d_plot(metric_x, metric_y, df, **kwargs):
    metric_x_unique_sorted = np.sort(df[metric_x].unique())
    min_x = metric_x_unique_sorted[0]
    if min_x > 0:
        metric_x_unique_sorted = np.concatenate((np.arange(0, min_x, 1), metric_x_unique_sorted))
    metric_y_unique_sorted = np.sort(df[metric_y].unique())
    
    Z = np.full((len(metric_x_unique_sorted), len(metric_y_unique_sorted)), np.nan)

    for ix, val_x in enumerate(metric_x_unique_sorted):
        for iy, val_y in enumerate(metric_y_unique_sorted):
            if kwargs.get('show_time_spent_LU', False):
                lu = df[(df[metric_x] == val_x) & (df[metric_y] == val_y)]['lu_reform'].mean()
                total = df[(df[metric_x] == val_x) & (df[metric_y] == val_y)]['t_reform'].mean()
                Z[ix, iy] = lu/total
            elif kwargs.get('show_flops', False):
                full = df[(df[metric_x] == val_x) & (df[metric_y] == val_y)]['flops_full'].mean()
                blocked = df[(df[metric_x] == val_x) & (df[metric_y] == val_y)]['flops_blocked'].mean()
                Z[ix,iy] = (blocked - full) / full
            else:
                full = df[(df[metric_x] == val_x) & (df[metric_y] == val_y)]['time_full'].mean()
                blocked = df[(df[metric_x] == val_x) & (df[metric_y] == val_y)]['time_blocked'].mean()
                Z[ix,iy] = (blocked - full) / full

    ax = kwargs.get("ax", None)
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    else:
        fig = ax.get_figure()

    if kwargs.get("simple_colorbar", False):
        cmap = 'gist_rainbow'
        norm = None
    else:
        level_jump = 0.1
        max_val = kwargs.get("max_val", None)
        if max_val is None:
            max_val = max(np.nanmax(Z), -np.nanmin(Z))
            max_val = min(1.1, max_val)
        levels = np.arange(level_jump*(np.nanmin(Z)//level_jump-1), level_jump*(np.nanmax(Z)//level_jump+1), level_jump)
        
        cmap = plt.cm.seismic

        # discrete colorbar with good scaling
        if kwargs.get('show_time_spent_LU', False):
            levels = np.arange(0, 1, level_jump)
        else:
            levels = np.arange(level_jump*(-max_val//level_jump), level_jump*(max_val//level_jump+2), level_jump)
        norm = mcolors.BoundaryNorm(levels, ncolors=256)
    
    # color nan values in Z in grey
    cmap.set_bad(color='lightgrey')
    
    # cntr = ax.contourf(metric_x_unique_sorted, metric_y_unique_sorted, Z, levels=levels, cmap=cmap, norm=norm)
    # cbar = fig.colorbar(cntr, ax=ax, label='Relative difference')
    mesh = ax.pcolormesh(metric_x_unique_sorted, metric_y_unique_sorted, Z.T, cmap=cmap, norm=norm)
    if kwargs.get('show_time_spent_LU', False):
        cbar = fig.colorbar(mesh, ax=ax, label='Percentage of time\nspent in LU')
    else:
        cbar = fig.colorbar(mesh, ax=ax, label='Relative difference', )

    ax.set_xlabel(metric_x)
    ax.set_ylabel(metric_y)
    plt.tight_layout()


def visualize_lu_scaling_2d(data):
    metrics = ['nu', 'nx', 'ng']

    for i in range(len(metrics)):
        for j in range(i):
            x = metrics[i]
            y = metrics[j]
            create_2d_plot(x, y, pd.DataFrame(data), show_time_spent_LU=True)
            plt.savefig(f'unittest/implicit/figures_benelux/lu_scaling_2d_{x}_{y}.png', dpi=300)
            
def check_out_parameter_distributions():
    file1 = 'build_docker/random_benchmark_results_huge.csv'
    file2 = 'build_docker/random_benchmark_results_extended.csv'

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    params = ['K', 'nx', 'r', 'nu', 'ng', 'ng_ineq']

    # for each parameter, plot normalized histograms for both files (in a figure per parameter)
    for param in params:
        plt.figure()
        plt.hist(df1[param], bins=20, alpha=0.5, label='Huge', density=True)
        plt.hist(df2[param], bins=20, alpha=0.5, label='Extended', density=True)
        plt.xlabel(param)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {param}')
        plt.legend()
        plt.grid()
        # plt.savefig(f'unittest/implicit/figures_benelux/distribution_{param}.png', dpi=300)
    plt.show()

check_out_parameter_distributions()
exit()

data = get_data()

# add flops
data = add_flops_to_data(data)
print("added flops to data")

# visualize_preprocessing_scaling(data)

visualize_scaling(data, show_flops=False, show_std=False, single_figure=True)
visualize_scaling(data, relative=True, show_flops=True, single_figure=True)

visualize_scaling_2d(data)
visualize_scaling_2d(data, x='nx', y='nu')
visualize_scaling_2d(data, x='nu', y='ng')

if 't_accel' in data.keys():
    visualize_scaling_2d(data, method='accel')
    visualize_scaling_2d(data, method='accel', x='nx', y='nu')
    visualize_scaling_2d(data, method='accel', x='nu', y='ng')

visualize_scaling_2d(data, show_flop=True)
visualize_scaling_2d(data, x='nx', y='nu', show_flop=True)
visualize_scaling_2d(data, x='nu', y='ng', show_flop=True)

# visualize_lu_scaling(data)
# visualize_lu_scaling(data, relative=True)

visualize_lu_scaling_2d(data)
