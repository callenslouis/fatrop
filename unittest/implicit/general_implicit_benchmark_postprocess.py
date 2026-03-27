import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random as rnd

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
    else:
        raise ValueError(f'Unknown label: {label}')

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
        'lu_impl': np.array(df['lu_impl'].values),
        'lu_reform': np.array(df['lu_reform'].values),
        'impl_decomp': np.array(df['impl_decomp'].values),
    }

    return data

def preprocessing(nx, nu, r, ng):
    # return 0.5*(nu + nx)*(nu + nx + 1)*(2*nx - 1) + \
    #     (nu + nx + 1)*nx*(2*nx - 1) + \
    #     0.5*(nu + nx)*(nu + nx + 1)*(2*nx - 1)
    return (2*nx + nu + 1)*(nx**2 + r**2) + \
                    (nx-r)*r*(5*nx + 3*nu + ng + 1)

def postprocessing(nx, nu, rho):
    # return nx*(2*nu - 1) + nx*(2*nx - 1) + nx*(2*nx - 1)
    return (2*nx - rho)*nx + rho**2 + nx**2

def backwardrecursion(nx, nu, nxp, ngp, ngi, rho, gamma, implicit=False):
    # w = np.min(gamma, nu + nx + 1)

    # w = gamma
    # return nxp*(nu + nx + 1)*(2*nxp - 1) + \
    #     0.5*(nu + nx)*(nu + nx + 1)*(2*nxp - 1) + \
    #     (nu + nx + 1)*ngp*(2*nx - 1) + \
    #     0.5*(nu + nx)*(nu + nx + 1)*(2*ngi - 1) + \
    #     2*w**3/3 - 2*w/3 + \
    #     (nu - rho + nx + 1)*(nu + nx)*(2*rho - 1) + \
    #     0.5*(nu - rho + nx - 1)*(nu + nx - rho)*(2*rho - 1) + \
    #     (nu - rho)**3/3 + \
    #     0.5*(nx + 1)*nx*(2*(nu - rho) - 1)

    flops = 2*(nu + nx + 1)*nx**2 + \
        (nu + nx + 1)*(nu + nx)*nx + \
        2*(nu + nx + 1)*gamma*nx + \
        2*(nu + nx + 1)*gamma*nx + \
        0.5*gamma*(nu + nx + 1)*min(nu, gamma) + \
        2*(nu - rho + nx + 1)*(nu + nx)*rho + \
        2*(nu - rho + nx + 1)*(nu + nx)*rho + \
        (nu - rho + nx + 1)*(nu + nx - rho)*rho + \
        (nu - rho)**3 + 0.5*(nu - rho)**2*(nx+1) + \
        (nx + 1)*nx*(nu-rho)
    if implicit:
        flops += 2*(nu + nx + 1)*(nu+nx)*nx + 2*(nu + nx)**2*nx + \
            2*(nu+nx)*((nu-rho)*rho +  nx*rho + rho) + \
            (nu+nx)*(nu-rho)*(0.5*(nu-rho) + 2*(nu+nx) + 2 + 2*nx)
    return flops

def backwardrecursion_initial_stage(nx, nu, gamma, rho):
    # w = min(gamma, nu + nx + 1)
    w = gamma
    return 2*w**3/3 - 2*w/3 + \
        (nx - rho + 1)*nx*(2*rho - 1) + \
        0.5*(nx - rho)*(nx - rho + 1)*(2*rho - 1) + \
        (nx - rho)**3/3 - (nx - rho)/3

def forwardrecursion_initial_stage(nx, rho):
    return (nx - rho)*(2*rho - 1) + nx*(2*rho - 1)

def forwardrecursion(nx, nu, nxp, ngi, rho, rhop, gammap, implicit=False):
    # return nx*(2*(nu - rho) - 1) + \
    #     (nx + nu - rho)*(2*rho - 1) + \
    #     (nu + nx)*(2*rho - 1) + \
    #     (nu + nx)*(2*ngi - 1) + \
    #     (nu + nx)*(2*nxp - 1) + \
    #     nxp*(2*nxp - 1) + \
    #     (gammap - rhop)*(2*nxp - 1)
    nux = nu + nx
    nur = nu - rho
    return 2*nx*nur + \
        nur**2 + \
        (nux*nur*3 if implicit else 0) + \
        2*(nux-rho)*rho + \
        2*nux*rho + \
        (2*nux*rho if implicit else 0) + \
        rho**2 + rho*gammap + \
        2*nux*nx + 2*nx**2 + 2*(gammap-rho)*nx + 2*nux*nx
    

def get_rough_flop_count(K, nx, nu, ng, r, **kwargs): 
    ngi = 0
    reformulated = kwargs.get('reformulated', False)
    implicit = kwargs.get('implicit', True)
    add_constant_offset = kwargs.get('add_constant_offset', True)

    nx = nx.copy(); nu = nu.copy(); ng = ng.copy(); r = r.copy()

    # if not isinstance(nu, int):
    #     nx = nx.copy(); nu = nu.copy(); ng = ng.copy(); r = r.copy()
    #     nu = nu[0]
    #     nx = nx[0]
    #     ng = ng[0]
    #     r = r[0]
    
    # set rho as a random number smaller than min(nu, ng)
    # rho = rnd.randint(0, min(nu, ng)-1)

    all_flops = []
    for i in range(len(nu)):
        # rho = ng[i]
        if min(nu[i], ng[i]) <= 1:
            rho = 0
        else:
            rho = rnd.randint(0, min(nu[i], ng[i])-1)
        # rho = min(nu[i], ng[i])
        # rho = 0
        if reformulated:
            ng[i] += nx[i]
            nu[i] += nx[i]
            flops = backwardrecursion(nx[i], nu[i], nx[i], ng[i], ngi, rho, ng[i]) + \
                    forwardrecursion(nx[i], nu[i], nx[i], ngi, rho, rho, ng[i])
            flops = K[i]*flops + 500000*add_constant_offset + \
                    backwardrecursion_initial_stage(nx[i], nu[i], ng[i], rho) + \
                    forwardrecursion_initial_stage(nx[i], rho)
            
        elif implicit:
            flops = backwardrecursion(r[i], nu[i] + (nx[i]-r[i]), r[i], ng[i], ngi, rho, ng[i], True) + \
                    forwardrecursion(r[i], nu[i] + (nx[i]-r[i]), r[i], ngi, rho, rho, ng[i], True)
            flops  += preprocessing(nx[i], nu[i], r[i], ng[i]) + \
                    postprocessing(nx[i], nu[i], nx[i])
            flops = K[i]*flops + 500000*add_constant_offset + \
                    backwardrecursion_initial_stage(r[i], nu[i] + (nx[i]-r[i]), ng[i], rho) + \
                    forwardrecursion_initial_stage(r[i], rho)
        
        else:
            flops = backwardrecursion(nx[i], nu[i], nx[i], ng[i], ngi, rho, ng[i]) + \
                    forwardrecursion(nx[i], nu[i], nx[i], ngi, rho, rho, ng[i])
            flops = K[i]*flops + 500000*add_constant_offset + \
                    backwardrecursion_initial_stage(nx[i], nu[i], ng[i], rho) + \
                    forwardrecursion_initial_stage(nx[i], rho)
        
        all_flops.append(flops)

    # return K*flops + 0*backwardrecursion_initial_stage(nx, nu, ng, ng) + \
    #           0*forwardrecursion_initial_stage(nx, ng) + 500000*add_constant_offset
    # return np.array([all_flops]) + 500000*add_constant_offset
    return np.array(all_flops)

def get_preprocessing_flop_count(data):
    nx = data['nx']
    nu = data['nu']
    r = data['r']
    ng = data['ng']
    K = data['K']

    # flop count for implicit preprocessing
    # flop_impl_pre = (2*nx + nu + 1)*(nx**2 + r**2) + \
    #                 nxmr*r*(5*nx + 3*nu + ng + 1)
    flop_impl_pre = preprocessing(nx, nu, r, ng)

    return K * flop_impl_pre

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
    metrics = ['K', 'nx', 'r', 'nu', 'ng']
    # metrics = ['nx']

    color_explicit = 'r'
    color_implicit = 'b'
    color_reformulated = 'g'

    if kwargs.get('show_flops', True):
        impl_flop = 0.2*get_rough_flop_count(data['K'], data['nx'], data['nu'], data['ng'], data['r'], reformulated=False, implicit=True)
        reform_flop = 0.2*get_rough_flop_count(data['K'], data['nx'], data['nu'], data['ng'], data['r'], reformulated=True, implicit=False)
        expl_flop = 0.2*get_rough_flop_count(data['K'], data['nx'], data['nu'], data['ng'], data['r'], reformulated=False, implicit=False)

    # for each metric, plot the scaling of the times
    for metric in metrics:
        plt.figure(figsize=(6,3.2))
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
                plt.plot(unique_sorted_metric, relative_flop, '-', alpha=0.5, label='FLOP estimate', color=color_implicit)
                plt.plot(unique_sorted_metric, relative_flop_expl, '-', alpha=0.5, label='FLOP estimate', color=color_explicit)

            plt.plot(unique_sorted_metric, expl_means, label='Explicit (relative)', color=color_explicit)
            # plt.fill_between(unique_sorted_metric, np.array(expl_means) - np.array(expl_stds)/np.array(reform_means), np.array(expl_means) + np.array(expl_stds)/np.array(reform_means), alpha=0.2, color=color_explicit)
            plt.plot(unique_sorted_metric, impl_means, label='Implicit (relative)', color=color_implicit)
            # plt.fill_between(unique_sorted_metric, np.array(impl_means) - np.array(impl_stds)/np.array(reform_means), np.array(impl_means) + np.array(impl_stds)/np.array(reform_means), alpha=0.2, color=color_implicit)
            plt.axhline(0, color='k', linestyle='-')
            plt.ylabel('Relative time difference')
        else:    
            if kwargs.get('show_flops', True):
                plt.plot(unique_sorted_metric, impl_flop_means, '-', alpha=0.5, label='FLOP estimate', color=color_implicit)
                plt.plot(unique_sorted_metric, expl_flop_means, '-', alpha=0.5, label='FLOP estimate', color=color_explicit)
                plt.plot(unique_sorted_metric, reform_flop_means, '-', alpha=0.5, label='FLOP estimate', color=color_reformulated)

            plt.plot(unique_sorted_metric, expl_means, label='Explicit', color=color_explicit)
            plt.fill_between(unique_sorted_metric, np.array(expl_means) - np.array(expl_stds), np.array(expl_means) + np.array(expl_stds), alpha=0.2, color=color_explicit)
            plt.plot(unique_sorted_metric, impl_means, label='Implicit', color=color_implicit)
            plt.fill_between(unique_sorted_metric, np.array(impl_means) - np.array(impl_stds), np.array(impl_means) + np.array(impl_stds), alpha=0.2, color=color_implicit)
            plt.plot(unique_sorted_metric, impl_pre_means, ':', label='Implicit pre', color='k')
            plt.plot(unique_sorted_metric, np.sum([impl_pre_means, impl_solve_means], axis=0), '--', label='Implicit solve', color='k')
            # plt.plot(unique_sorted_metric, np.sum([impl_pre_means, impl_solve_means, impl_post_means], axis=0), ':', label='Implicit post')
            plt.plot(unique_sorted_metric, reform_means, label='Reformulation', color=color_reformulated)
            plt.fill_between(unique_sorted_metric, np.array(reform_means) - np.array(reform_stds), np.array(reform_means) + np.array(reform_stds), alpha=0.2, color=color_reformulated)
            plt.ylabel('Time (ns)')

        plt.xlabel(translate_label(metric))
        # plt.title(f'Scaling of Times with {metric}')
        plt.legend()
        plt.grid()
        # plt.xscale('log')
        # plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'unittest/implicit/figures_benelux/scaling_{metric}{"_relative" if kwargs.get("relative", False) else ""}{"_flops" if kwargs.get("show_flops") else ""}.png', dpi=300)

def visualize_scaling_2d(data, **kwargs):
    x = kwargs.get('x', 'nx')
    y = kwargs.get('y', 'ng')
    x_unique = np.sort(np.unique(data[x]))
    y_unique = np.sort(np.unique(data[y]))

    Z = np.full((len(y_unique), len(x_unique)), np.nan)

    if kwargs.get('show_flop', False):
        impl_flop = get_rough_flop_count(data['K'], data['nx'], data['nu'], data['ng'], data['r'], reformulated=False, implicit=True)
        reform_flop = get_rough_flop_count(data['K'], data['nx'], data['nu'], data['ng'], data['r'], reformulated=True, implicit=False)

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
                Z[i, j] = (mean_impl - mean_reform) / mean_reform

    fig, ax = plt.subplots()
    norm = mcolors.TwoSlopeNorm(vmin=min(-0.01, np.nanmin(Z)), vcenter=0, vmax=max(np.nanmax(Z),0.01))
    mesh = ax.pcolormesh(x_unique, y_unique, Z, cmap='bwr', norm=norm)
    fig.colorbar(mesh, ax=ax, label='Relative difference')
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
        plt.show()

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
    
    plt.show()

data = get_data()

# visualize_preprocessing_scaling(data)

visualize_scaling(data, show_flop=True)
visualize_scaling(data, relative=True, show_flops=False)
visualize_scaling_2d(data)
visualize_scaling_2d(data, x='nx', y='nu')
visualize_scaling_2d(data, x='nu', y='ng')
# visualize_scaling_2d(data, show_flop=True)

# visualize_lu_scaling(data)
# visualize_lu_scaling(data, relative=True)
visualize_lu_scaling_2d(data)
