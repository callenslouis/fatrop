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
              'axes.labelsize': 20,
              'axes.titlesize': 15,
              'legend.fontsize': 15,
              'xtick.labelsize': 15,
              'ytick.labelsize': 15,
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
    file = 'build_docker/random_benchmark_results.csv'
    # file = 'build_docker/random_benchmark_results_benelux.csv'
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
        'lu_impl': np.array(df['lu_impl'].values),
        'lu_reform': np.array(df['lu_reform'].values),
        'impl_decomp': np.array(df['impl_decomp'].values),
    }

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

    if kwargs.get('show_flops', True):
        fc = FlopCounter()
        impl_flop, expl_flop, reform_flop = fc.get_all_flops(data)

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
        expl_means = []; impl_means = []; reform_means = []
        expl_stds = []; impl_stds = []; reform_stds = []
        impl_pre_means = []; impl_pre_stds = []
        impl_solve_means = []; impl_solve_stds = []
        impl_post_means = []; impl_post_stds = []
        
        impl_flop_means = []; reform_flop_means = []; expl_flop_means = []

        for val in unique_sorted_metric:
            mask = data[metric] == val
            expl_means.append(np.mean(data['t_expl'][mask]))
            expl_stds.append(np.std(data['t_expl'][mask]))
            impl_means.append(np.mean(data['t_impl'][mask]))
            impl_stds.append(np.std(data['t_impl'][mask]))
            reform_means.append(np.mean(data['t_reform'][mask]))
            reform_stds.append(np.std(data['t_reform'][mask]))

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

        if kwargs.get('relative', False):
            expl_means = (np.array(expl_means) - np.array(reform_means)) / np.array(reform_means)
            impl_means = (np.array(impl_means) - np.array(reform_means)) / np.array(reform_means)
            if kwargs.get('show_flops', True):
                relative_flop = (np.array(impl_flop_means) - np.array(reform_flop_means)) / np.array(reform_flop_means)
                relative_flop_expl = (np.array(expl_flop_means) - np.array(reform_flop_means)) / np.array(reform_flop_means)
                ax.plot(unique_sorted_metric, relative_flop, '-', alpha=0.5, label='FLOP estimate', color=color_implicit)
                ax.plot(unique_sorted_metric, relative_flop_expl, '-', alpha=0.5, label='FLOP estimate', color=color_explicit)

            ax.plot(unique_sorted_metric, expl_means, label='Explicit (relative)', color=color_explicit)
            # plt.fill_between(unique_sorted_metric, np.array(expl_means) - np.array(expl_stds)/np.array(reform_means), np.array(expl_means) + np.array(expl_stds)/np.array(reform_means), alpha=0.2, color=color_explicit)
            ax.plot(unique_sorted_metric, impl_means, label='Implicit (relative)', color=color_implicit)
            # plt.fill_between(unique_sorted_metric, np.array(impl_means) - np.array(impl_stds)/np.array(reform_means), np.array(impl_means) + np.array(impl_stds)/np.array(reform_means), alpha=0.2, color=color_implicit)
            ax.axhline(0, color='k', linestyle='-')
            ax.set_ylabel('Relative time difference')
        else:    
            if kwargs.get('show_flops', True):
                ax.plot(unique_sorted_metric, impl_flop_means, '-', alpha=0.5, label='FLOP estimate', color=color_implicit)
                ax.plot(unique_sorted_metric, expl_flop_means, '-', alpha=0.5, label='FLOP estimate', color=color_explicit)
                ax.plot(unique_sorted_metric, reform_flop_means, '-', alpha=0.5, label='FLOP estimate', color=color_reformulated)

            ax.plot(unique_sorted_metric, expl_means, label='Explicit', color=color_explicit)
            ax.fill_between(unique_sorted_metric, np.array(expl_means) - np.array(expl_stds), np.array(expl_means) + np.array(expl_stds), alpha=0.2, color=color_explicit)
            ax.plot(unique_sorted_metric, impl_means, label='Implicit', color=color_implicit)
            ax.fill_between(unique_sorted_metric, np.array(impl_means) - np.array(impl_stds), np.array(impl_means) + np.array(impl_stds), alpha=0.2, color=color_implicit)
            ax.plot(unique_sorted_metric, impl_pre_means, ':', label='Implicit pre', color='k')
            ax.plot(unique_sorted_metric, np.sum([impl_pre_means, impl_solve_means], axis=0), '--', label='Implicit solve', color='k')
            # ax.plot(unique_sorted_metric, np.sum([impl_pre_means, impl_solve_means, impl_post_means], axis=0), ':', label='Implicit post')
            ax.plot(unique_sorted_metric, reform_means, label='Reformulation', color=color_reformulated)
            ax.fill_between(unique_sorted_metric, np.array(reform_means) - np.array(reform_stds), np.array(reform_means) + np.array(reform_stds), alpha=0.2, color=color_reformulated)
            ax.set_ylabel('Time (ns)')

        ax.set_xlabel(translate_label(metric))
        # plt.title(f'Scaling of Times with {metric}')
        ax.grid()
        plt.tight_layout()
        if not kwargs.get('single_figure', False):
            ax.legend()
            plt.savefig(f'unittest/implicit/figures_benelux/scaling_{metric}{"_relative" if kwargs.get("relative", False) else ""}{"_flops" if kwargs.get("show_flops") else ""}.png', dpi=300)
    
    if kwargs.get('single_figure', False):
        plt.savefig(f'unittest/implicit/figures_benelux/scaling_single_figure{"_relative" if kwargs.get("relative", False) else ""}{"_flops" if kwargs.get("show_flops") else ""}.png', dpi=300)

def visualize_scaling_2d(data, **kwargs):
    x = kwargs.get('x', 'nx')
    y = kwargs.get('y', 'ng')
    x_unique = np.sort(np.unique(data[x]))
    y_unique = np.sort(np.unique(data[y]))

    Z = np.full((len(y_unique), len(x_unique)), np.nan)

    if kwargs.get('show_flop', False):
        # impl_flop = get_rough_flop_count(data['K'], data['nx'], data['nu'], data['ng'], data['r'], 
        # reformulated=False, implicit=True)
        # reform_flop = get_rough_flop_count(data['K'], data['nx'], data['nu'], data['ng'], data['r'], reformulated=True, implicit=False)
        fc = FlopCounter()
        impl_flop, expl_flop, reform_flop = fc.get_all_flops(data)

        data_x = impl_flop
        data_y = reform_flop
    else:
        data_x = data['t_impl']
        data_y = data['t_reform']

    for i, y_val in enumerate(y_unique):
        for j, x_val in enumerate(x_unique):
            mask = (data[x] == x_val) & (data[y] == y_val)
            if mask.any():
                mean_impl = np.mean(data_x[mask])
                mean_reform = np.mean(data_y[mask])
                if kwargs.get('relative', True):
                    Z[i, j] = (mean_impl - mean_reform) / mean_reform
                else:
                    Z[i, j] = (mean_impl - mean_reform)

    fig, ax = plt.subplots()
    norm = mcolors.TwoSlopeNorm(vmin=min(-0.01, np.nanmin(Z)), vcenter=0, vmax=max(np.nanmax(Z),0.01))
    mesh = ax.pcolormesh(x_unique, y_unique, Z, cmap='bwr', norm=norm)
    cbar = fig.colorbar(mesh, ax=ax, label='Relative difference')
    ticks = np.linspace(norm.vmin, norm.vmax, 7)  # or choose number you like
    cbar.set_ticks(ticks)
    ax.set_xlabel(translate_label(x))
    ax.set_ylabel(translate_label(y))
    # ax.set_title('Scaling of Implicit Time with nx and ng')
    # plt.show()
    plt.tight_layout()
    plt.savefig(f'unittest/implicit/figures_benelux/scaling_2d_{x}_{y}{"_relative" if kwargs.get("relative", False) else ""}{"_flops" if kwargs.get("show_flop") else ""}.png', dpi=300)
    

def visualize_lu_scaling(data, **kwargs):
    metrics = ['nx', 'nu', 'ng']

    for metric in metrics:
        plt.figure()
        unique_metric = np.unique(data[metric])
        unique_sorted_metric = np.sort(unique_metric)

        if kwargs.get('relative', False):
            lu_impl_means = []
            lu_reform_means = []

            for val in unique_sorted_metric:
                mask = data[metric] == val
                lu_impl_means.append(np.mean(data['lu_impl'][mask]) / np.mean(data['lu_reform'][mask]))
                lu_reform_means.append(1.0)

            plt.plot(unique_sorted_metric, lu_impl_means, label='Implicit / Reformulation')
            plt.ylabel('Relative LU solve time')
        else:
            lu_impl_means = []
            lu_impl_stds = []
            lu_reform_means = []
            lu_reform_stds = []

            for val in unique_sorted_metric:
                mask = data[metric] == val
                lu_impl_means.append(np.mean(data['lu_impl'][mask]))
                lu_impl_stds.append(np.std(data['lu_impl'][mask]))
                lu_reform_means.append(np.mean(data['lu_reform'][mask]))
                lu_reform_stds.append(np.std(data['lu_reform'][mask]))

            plt.plot(unique_sorted_metric, lu_impl_means, label='Implicit')
            plt.plot(unique_sorted_metric, lu_reform_means, label='Reformulation')
            plt.fill_between(unique_sorted_metric, np.array(lu_impl_means) - np.array(lu_impl_stds), np.array(lu_impl_means) + np.array(lu_impl_stds), alpha=0.2)
            plt.fill_between(unique_sorted_metric, np.array(lu_reform_means) - np.array(lu_reform_stds), np.array(lu_reform_means) + np.array(lu_reform_stds), alpha=0.2)
            plt.ylabel('LU Solve Time (s)')
        
        plt.xlabel(metric)
        plt.title(f'Scaling of LU Solve Time with {metric}')
        plt.legend()
        plt.grid()
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()

def visualize_lu_scaling_2d(data):
    # # x-axis: nx, y-axis: nu, color: (lu_impl - lu_reform) / lu_reform
    # plt.figure()
    # x = data['nx']
    # y = data['nu']
    # # color = (data['lu_impl'] - data['lu_reform']) / data['lu_reform']
    # color = data['lu_reform']
    # # color = data['lu_impl']
    # plt.scatter(x, y, c=color, cmap='coolwarm', edgecolor=None)
    # plt.colorbar(label='Relative LU solve time difference')
    # plt.xlabel(translate_label('nx'))
    # plt.ylabel(translate_label('nu'))
    # plt.title('Relative LU Solve Time Difference (Implicit vs Reformulation)')
    # plt.grid()

    # # expected relative difference (-nu*nx^2/(nu*nx^2 + nx^3))
    # plt.figure()
    # x = data['nx']
    # y = data['nu']
    # # expected_color = (-y*x**2) / (y*x**2 + x**3)
    # expected_color = (y*x**2 + x**3)
    # # expected_color = x**3
    # plt.scatter(x, y, c=expected_color, cmap='coolwarm', edgecolor=None)
    # plt.colorbar(label='Expected Relative LU solve time difference')
    # plt.xlabel(translate_label('nx'))
    # plt.ylabel(translate_label('nu'))
    # plt.title('Expected Relative LU Solve Time Difference')
    
    # plt.show()




    # # plt.figure()
    # x = data['nx']
    # y = data['nu']
    # # color = (data['lu_impl'] - data['lu_reform']) / data['lu_reform']
    # # color = data['lu_reform']
    # color = data['lu_impl']
    # plt.scatter(x, y, c=color, cmap='coolwarm', edgecolor=None)
    # plt.colorbar(label='Relative LU solve time difference')
    # plt.xlabel(translate_label('nx'))
    # plt.ylabel(translate_label('nu'))
    # plt.title('Relative LU Solve Time Difference (Implicit vs Reformulation)')
    # plt.grid()

    # # expected relative difference (-nu*nx^2/(nu*nx^2 + nx^3))
    # plt.figure()
    # x = data['nx']
    # y = data['nu']
    # # expected_color = (-y*x**2) / (y*x**2 + x**3)
    # # expected_color = (y*x**2 + x**3)
    # expected_color = x**3
    # plt.scatter(x, y, c=expected_color, cmap='coolwarm', edgecolor=None)
    # plt.colorbar(label='Relative difference')
    # plt.xlabel(translate_label('nx'))
    # plt.ylabel(translate_label('nu'))
    # # plt.title('Expected Relative LU Solve Time Difference')
    # plt.tight_layout()
    # plt.savefig("unittest/implicit/figures_benelux/scaling_2d_lu_impl_expected.png", dpi=300)
    
    # plt.show()




    # # plt.figure()
    # # x = data['nx']
    # # y = data['nu']
    # # # color = (data['lu_impl'] - data['lu_reform']) / data['lu_reform']
    # # # color = data['lu_reform']
    # # # color = data['lu_impl']
    # # color = data['impl_decomp']
    # # plt.scatter(x, y, c=color, cmap='coolwarm', edgecolor=None)
    # # plt.colorbar(label='Relative LU solve time difference')
    # # plt.xlabel('nx')
    # # plt.ylabel('nu')
    # # plt.title('Relative LU Solve Time Difference (Implicit vs Reformulation)')
    # # plt.grid()

    # # # expected relative difference (-nu*nx^2/(nu*nx^2 + nx^3))
    # # plt.figure()
    # # x = data['nx']
    # # y = data['nu']
    # # # expected_color = (-y*x**2) / (y*x**2 + x**3)
    # # # expected_color = (y*x**2 + x**3)
    # # expected_color = x**3
    # # plt.scatter(x, y, c=expected_color, cmap='coolwarm', edgecolor=None)
    # # plt.colorbar(label='Expected Relative LU solve time difference')
    # # plt.xlabel('nx')
    # # plt.ylabel('nu')
    # # plt.title('Expected Relative LU Solve Time Difference')
    
    # # plt.show()


    # plt.figure()
    # x = data['nx']
    # y = data['nu']
    # color = (data['lu_impl'] - data['lu_reform']) / data['lu_reform']
    # plt.scatter(x, y, c=color, cmap='coolwarm', edgecolor=None)
    # plt.colorbar(label='Relative LU solve time difference')
    # plt.xlabel('nx')
    # plt.ylabel('nu')
    # plt.title('Relative LU Solve Time Difference (Implicit vs Reformulation)')
    # plt.grid()

    # expected relative difference (-nu*nx^2/(nu*nx^2 + nx^3))
    plt.figure()
    x = data['nx']
    y = data['ng']
    expected_color = (-y*x**2) / (y*x**2 + x**3)
    plt.scatter(x, y, c=expected_color, cmap='coolwarm', edgecolor=None)
    plt.colorbar(label='Expected Relative LU solve time difference')
    plt.xlabel('nx')
    plt.ylabel('ng')
    plt.title('Expected Relative LU Solve Time Difference')
    
    # plt.show()

data = get_data()

# visualize_preprocessing_scaling(data)

visualize_scaling(data, show_flop=True, single_figure=True)
visualize_scaling(data, relative=True, show_flops=True, single_figure=True)

visualize_scaling_2d(data)
visualize_scaling_2d(data, x='nx', y='nu')
visualize_scaling_2d(data, x='nu', y='ng')

visualize_scaling_2d(data, show_flop=True)
visualize_scaling_2d(data, x='nx', y='nu', show_flop=True)
visualize_scaling_2d(data, x='nu', y='ng', show_flop=True)

# visualize_lu_scaling(data)
# visualize_lu_scaling(data, relative=True)
# visualize_lu_scaling_2d(data)
