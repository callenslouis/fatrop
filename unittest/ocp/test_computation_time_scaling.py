import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import sys
sys.path.append('build/unittest/')
from scaling_test_results_explicit import *
from scaling_test_results_reformulation import *
from scaling_test_results_implicit import *

def preprocessing(nx, nu, nxp):
    return nxp*(nx + nu + 1)*(2*nxp - 1) + \
        (nx + nu)*(nx + nu + 1)*(4*nx - 1)

def postprocessing(nx, nu, nxp):
    return nxp*(2*nu - 1) + nxp*(2*nx - 1) + nxp*(2*nxp - 1)

def backwardrecursion(nx, nu, nxp, ngp, ngi, rho, gamma):
    # w = min(gamma, nu + nx + 1)
    w = gamma
    return nxp*(nu + nx + 1)*(2*nxp - 1) + \
        0.5*(nu + nx)*(nu + nx + 1)*(2*nxp - 1) + \
        (nu + nx + 1)*ngp*(2*nx - 1) + \
        0.5*(nu + nx)*(nu + nx + 1)*(2*ngi - 1) + \
        2*w**3/3 - 2*w/3 + \
        (nu - rho + nx + 1)*(nu + nx)*(2*rho - 1) + \
        0.5*(nu - rho + nx - 1)*(nu + nx - rho)*(2*rho - 1) + \
        (nu - rho)**3/3 + \
        0.5*(nx + 1)*nx*(2*(nu - rho) - 1)

def forwardrecursion(nx, nu, nxp, ngi, rho, rhop, gammap):
    return nx*(2*(nu - rho) - 1) + \
        (nx + nu - rho)*(2*rho - 1) + \
        (nu + nx)*(2*rho - 1) + \
        (nu + nx)*(2*ngi - 1) + \
        (nu + nx)*(2*nxp - 1) + \
        nxp*(2*nxp - 1) + \
        (gammap - rhop)*(2*nxp - 1)

def get_rough_flop_count(K, nx, nu, ng, **kwargs):
    implicit = kwargs.get('implicit', False)
    reformulated = kwargs.get('reformulated', False)
    if implicit and reformulated:
        raise ValueError("Cannot use both implicit and reformulated options together.")
    
    ngi = kwargs.get('ngi', 0.5*ng)

    if not isinstance(nu, int):
        nu = nu.copy()
        nx = nx.copy()
        ng = ng.copy()

    if reformulated:
        ng += nx
        nu += nx

    flops = backwardrecursion(nx, nu, nx, ng, ngi, ng, ng) + \
            forwardrecursion(nx, nu, nx, ngi, ng, ng, ng)
    if implicit:
        flops  += preprocessing(nx, nu, nx) + \
                  postprocessing(nx, nu, nx)

    return K*flops

def visualize_3d():
    nx = np.arange(1, 15)
    nu = np.arange(1, 7)
    ng = np.arange(1, 5)

    print(f"nx: {nx}")
    print(f"nu: {nu}")
    print(f"ng: {ng}")

    X, Y, Z = np.meshgrid(nx, nu, ng, indexing='ij')
    print(f"X: {X.flatten()}")
    print(f"Y: {Y.flatten()}")
    print(f"Z: {Z.flatten()}")

    W = get_rough_flop_count(X, Y, Z)
    W_implicit = get_rough_flop_count(X, Y, Z, implicit=True)
    W_implicit_overhead = W_implicit - W
    W_reformulated = get_rough_flop_count(X, Y, Z, reformulated=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ms = 1
    ax.scatter(X.flatten(), Y.flatten(), W.flatten(), c='r', marker='o', s=ms, label='Explicit')
    ax.scatter(X.flatten(), Y.flatten(), W_implicit.flatten(), c='b', marker='^', s=ms, label='Implicit')
    # ax.scatter(X.flatten(), Y.flatten(), W_implicit_overhead.flatten(), c='b', marker='^', s=ms, label='Implicit Overhead')
    ax.scatter(X.flatten(), Y.flatten(), W_reformulated.flatten(), c='g', marker='s', s=ms, label='Reformulated')

    plot_polynomial_surface(ax, X.flatten(), Y.flatten(), W.flatten(), color='r', label='Explicit')
    plot_polynomial_surface(ax, X.flatten(), Y.flatten(), W_implicit.flatten(), color='b', label='Implicit')
    # plot_polynomial_surface(ax, X.flatten(), Y.flatten(), W_implicit_overhead.flatten(), color='b', label='Implicit Overhead')
    plot_polynomial_surface(ax, X.flatten(), Y.flatten(), W_reformulated.flatten(), color='g', label='Reformulated')

    ax.set_xlabel('nx')
    ax.set_ylabel('nu')
    ax.set_zlabel('Flops')

    ax.legend()
    
    plt.show()

def get_average_Z(X, Y, Z):
    # for all pairs [x, y], compute the average z value and return
    # X_unique, Y_unique, Z_avg

    unique_pairs = {}
    for x, y, z in zip(X, Y, Z):
        key = (x, y)
        if key not in unique_pairs:
            unique_pairs[key] = []
        unique_pairs[key].append(z)

    X_unique = []
    Y_unique = []
    Z_avg = []
    for (x, y), zs in unique_pairs.items():
        X_unique.append(x)
        Y_unique.append(y)
        Z_avg.append(np.mean(zs))

    return np.array(X_unique), np.array(Y_unique), np.array(Z_avg)

def get_median_Z(X, Y, Z):
    unique_pairs = {}
    for x, y, z in zip(X, Y, Z):
        key = (x, y)
        if key not in unique_pairs:
            unique_pairs[key] = []
        unique_pairs[key].append(z)

    X_unique = []
    Y_unique = []
    Z_median = []
    for (x, y), zs in unique_pairs.items():
        X_unique.append(x)
        Y_unique.append(y)
        Z_median.append(np.median(zs))

    return np.array(X_unique), np.array(Y_unique), np.array(Z_median)

def fit_polynomial_surface(X, Y, Z):
    from scipy.optimize import curve_fit

    MAX_DEGREE = 2

    if MAX_DEGREE == 3:
        def polynomial_surface(xy, x1, x2, x3, y1, y2, y3, c):
            x, y = xy
            return x1*x + x2*x**2 + x3*x**3 + y1*y + y2*y**2 + y3*y**3 + c
        
        # Fit the polynomial surface
        popt, _ = curve_fit(polynomial_surface, (X, Y), Z)

        print(f"Surface fit: t(nx, nu) = {popt[0]}*nx + {popt[1]}*nx^2 + {popt[2]}*nx^3 + "
            f"{popt[3]}*nu + {popt[4]}*nu^2 + {popt[5]}*nu^3 + {popt[6]}")
        
    elif MAX_DEGREE == 2:
        def polynomial_surface(xy, x1, x2, y1, y2, c):
            x, y = xy
            return x1*x + x2*x**2 + y1*y + y2*y**2 + c
        
        # Fit the polynomial surface
        popt, _ = curve_fit(polynomial_surface, (X, Y), Z)

        print(f"Surface fit: t(nx, nu) = {popt[0]}*nx + {popt[1]}*nx^2 + "
            f"{popt[2]}*nu + {popt[3]}*nu^2 + {popt[4]}")

    # get surface Z values on the grid
    X_grid, Y_grid = np.meshgrid(np.unique(X), np.unique(Y), indexing='ij')
    Z_grid = polynomial_surface((X_grid, Y_grid), *popt)

    return popt, X_grid, Y_grid, Z_grid

def plot_polynomial_surface(ax, X, Y, Z, **kwargs):
    popt, X_grid, Y_grid, Z_grid = fit_polynomial_surface(X, Y, Z)
    kwargs["label"] = None
    if "color" in kwargs.keys():
        ax.plot_surface(X_grid, Y_grid, Z_grid, color=kwargs.get("color", 'k'), label=kwargs.get("label", None))
    else:
        ax.plot_surface(X_grid, Y_grid, Z_grid, label=kwargs.get("label", None))

def visualize_experiments(X, Y, Z, **kwargs):
    if 'ax' not in kwargs.keys():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = kwargs['ax']

    color = kwargs.get('color', 'b')
    show = kwargs.get('show', True)
    show_surface = kwargs.get('show_surface', True)
    ms = kwargs.get('marker_size', 1)

    XX, YY, ZZ = get_median_Z(X, Y, Z)
    ax.scatter(XX, YY, ZZ, c=color, marker='^', s=ms)
    if show_surface:
        plot_polynomial_surface(ax, XX, YY, ZZ, color=color)


    ax.set_xlabel('nx')
    ax.set_ylabel('nu')
    ax.set_zlabel('microseconds')
    ax.set_zlim([0, 250])

    if show:
        plt.tight_layout()
        plt.show()

def visualize_experiments_vs_flop_count(X, Y, Z):

    XX, YY, ZZ = get_median_Z(X, Y, Z)
    flop_counts = get_rough_flop_count(XX, YY, XX/2)

    ms_per_flop = ZZ / flop_counts

    # plot ms_per_flop as a heatplot
    fig = plt.figure()
    plt.scatter(XX, YY, c=np.log(ms_per_flop), cmap='viridis', s=15)
    plt.colorbar(label='microseconds per flop')
    ax = plt.gca()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ms = 1
    # ax.scatter(XX, YY, ms_per_flop, c='b', marker='^', s=ms)
    # # plot_polynomial_surface(ax, XX, YY, ms_per_flop, color='b')

    # print(ms_per_flop)

    ax.set_xlabel('nx')
    # ax.set_ylabel('nu')
    # ax.set_zlabel('microseconds per flop')

    plt.show()  

def plot_line_with_std(ax, x_vals, xx, time, color, alpha, x_vals_offsets=None, **kwargs):
    show_std = kwargs.get('show_std', True)
    lw = kwargs.get('linewidth', 2.0)
    marker = kwargs.get('marker', None)
    linestyle = kwargs.get('linestyle', '-')
    
    t_vals = 0*x_vals
    t_vals_min_std = 0*t_vals
    t_vals_plus_std = 0*t_vals

    if x_vals_offsets is None:
        x_vals_offsets = 0*x_vals

    MAX = 5000
    for i in range(len(x_vals)):
        mask = (xx == x_vals[i]) & (time < MAX)
        std = np.std(time[mask])
        t_vals[i] = np.median(time[mask])
        t_vals_min_std[i] = t_vals[i] - std
        t_vals_plus_std[i] = t_vals[i] + std

    orig_t_vals = t_vals.copy()
    t_vals += x_vals_offsets
    t_vals_min_std += x_vals_offsets
    t_vals_plus_std += x_vals_offsets
    x_vals_offsets += orig_t_vals

    if show_std:
        ax.fill_between(x_vals, t_vals_min_std, t_vals_plus_std, color=color, alpha=0.1)
    ax.plot(x_vals, t_vals, color=color, linewidth=lw, marker=marker, linestyle=linestyle)

    return max(t_vals_plus_std)


def show_2d_plot(K, nx, nu, ng, ng_ineq, time, **kwargs):
    color = kwargs.get('color', 'b')
    show = kwargs.get('show', True)
    show_details = "timing_details" in kwargs and kwargs['timing_details'] is not None
    independent_var = kwargs.get('independent_var', 'nx')
    if independent_var not in ['K', 'nx', 'nu', 'ng', 'ng_ineq']:
        raise ValueError("independent_var must be in ['K', 'nx', 'nu', 'ng', 'ng_ineq']")

    if 'ax' in kwargs.keys():
        ax = kwargs['ax']
    else:
        plt.figure()
        ax = plt.gca()

    if independent_var == 'K':
        x_vals = np.unique(K)
        xx = K
    elif independent_var == 'nx':
        x_vals = np.unique(nx)
        xx = nx
    elif independent_var == 'nu':
        x_vals = np.unique(nu)
        xx = nu
    elif independent_var == 'ng':
        x_vals = np.unique(ng)
        xx = ng
    elif independent_var == 'ng_ineq':
        x_vals = np.unique(ng_ineq)
        xx = ng_ineq

    max_y_val = -1
    y = plot_line_with_std(ax, x_vals, xx, time, color=color, alpha=0.1)
    max_y_val = max(max_y_val, y)

    if show_details:
        x_vals_offsets = 0*x_vals
        for detail_name, detail in kwargs['timing_details'].items():
            y = plot_line_with_std(ax, x_vals, xx, detail['time'], 
                                   color=detail['color'], alpha=0.1, 
                                   x_vals_offsets=x_vals_offsets, 
                                   show_std=False, linewidth=1.0,
                                   marker=detail.get('marker', None),
                                   linestyle=detail.get('linestyle', '-'))
            max_y_val = max(max_y_val, y)

    ax.set_xlabel(independent_var)
    # ax.set_ylabel('t_comp [us]')
    ax.set_xlim([np.min(x_vals), np.max(x_vals)])
    curr_y_lim = ax.get_ylim()
    ax.set_ylim([0, max(curr_y_lim[1], 1.1*max_y_val)])

    ylog_vars = ['nx', 'nu', 'ng']
    if independent_var in ylog_vars:
        ax.set_yscale('log')
        ax.set_ylim([10, max(curr_y_lim[1], 1.1*max_y_val)])

    if show:
        plt.tight_layout()
        plt.show()

def show_2d_plot_all_cases(**kwargs):
    if 'ax' in kwargs.keys():
        ax = kwargs['ax']
    else:
        plt.figure()
        ax = plt.gca()

    show = kwargs.get("show", False)
    iv = kwargs.get("independent_var", 'nx')
    implicit_details = kwargs.get("implicit_timings_details", None)

    show_2d_plot(K_reformulation, nx_reformulation, nu_reformulation, ng_reformulation, ng_ineq_reformulation, time_reformulation, ax=ax, color='g', show=False, independent_var=iv)
    show_2d_plot(K_implicit, nx_implicit, nu_implicit, ng_implicit, ng_ineq_implicit, time_implicit, ax=ax, color='b', show=show, independent_var=iv, timing_details=implicit_details)
    show_2d_plot(K_explicit, nx_explicit, nu_explicit, ng_explicit, ng_ineq_explicit, time_explicit, ax=ax, color='r', show=False, independent_var=iv)

def show_flops_2d_plot(K, nx, nu, ng, ng_ineq, time, **kwargs):
    color = kwargs.get('color', 'b')
    show = kwargs.get('show', True)
    independent_var = kwargs.get('independent_var', 'nx')
    if independent_var not in ['K', 'nx', 'nu', 'ng', 'ng_ineq']:
        raise ValueError("independent_var must be in ['K', 'nx', 'nu', 'ng', 'ng_ineq']")

    if 'ax' in kwargs.keys():
        ax = kwargs['ax']
    else:
        plt.figure()
        ax = plt.gca()

    if independent_var == 'K':
        x_vals = np.unique(K)
        xx = K
    elif independent_var == 'nx':
        x_vals = np.unique(nx)
        xx = nx
    elif independent_var == 'nu':
        x_vals = np.unique(nu)
        xx = nu
    elif independent_var == 'ng':
        x_vals = np.unique(ng)
        xx = ng
    elif independent_var == 'ng_ineq':
        x_vals = np.unique(ng_ineq)
        xx = ng_ineq
    
    T = 0*nx
    t_vals = 0*x_vals
    t_vals_min_std = 0*t_vals
    t_vals_plus_std = 0*t_vals

    for i in range(len(nx)):
        T[i] = get_rough_flop_count(K[i], nx[i], nu[i], ng[i], implicit=kwargs.get("implicit", False), reformulated=kwargs.get("reformulated", False))

    for i in range(len(x_vals)):
        mask = (xx == x_vals[i])
        std = np.std(T[mask])
        t_vals[i] = np.median(T[mask])
        t_vals_min_std[i] = t_vals[i] - std
        t_vals_plus_std[i] = t_vals[i] + std

    ax.fill_between(x_vals, t_vals_min_std, t_vals_plus_std, color=color, alpha=0.1)
    ax.plot(x_vals, t_vals, color=color)

    ax.set_xlabel(independent_var)
    # ax.set_ylabel('# flops')
    ax.set_xlim([np.min(x_vals), np.max(x_vals)])
    curr_y_lim = ax.get_ylim()
    ax.set_ylim([0, max(curr_y_lim[1], 1.1*np.max(t_vals_plus_std))])

    ylog_vars = ['nx', 'nu', 'ng']
    if independent_var in ylog_vars:
        ax.set_yscale('log')
        ax.set_ylim([1e5, max(curr_y_lim[1], 1.1*np.max(t_vals_plus_std))])

    if show:
        plt.tight_layout()
        plt.show()
    
def show_flops_2d_plot_all_cases(**kwargs):
    if 'ax' in kwargs.keys():
        ax = kwargs['ax']
    else:
        plt.figure()
        ax = plt.gca()

    show = kwargs.get("show", False)
    iv = kwargs.get("independent_var", 'nx')

    show_flops_2d_plot(K_explicit, nx_explicit, nu_explicit, ng_explicit, ng_ineq_explicit, time_explicit, ax=ax, color='r', show=False, independent_var=iv)
    show_flops_2d_plot(K_reformulation, nx_reformulation, nu_reformulation, ng_reformulation, ng_ineq_reformulation, time_reformulation, ax=ax, color='g', show=False, reformulated=True, independent_var=iv)
    show_flops_2d_plot(K_implicit, nx_implicit, nu_implicit, ng_implicit, ng_ineq_implicit, time_implicit, ax=ax, color='b', show=show, implicit=True, independent_var=iv)

def add_legend_below(fig, timing_details=None):
    # create some space below the axes
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25)

    # get basic labels
    labels = ['Explicit', 'Implicit', 'Reformulated']
    colors = ['r', 'b', 'g']
    handles = [plt.Line2D([0], [0], color=color, lw=2) for color in colors]


    # add detailed handles if provided
    if timing_details is not None:
        # add empty entry
        labels.append('')
        handles.append(plt.Line2D([0], [0], color='none', lw=0))

        # add new entries
        for detail_name, detail in timing_details.items():
            labels.append(detail_name)
            handles.append(plt.Line2D([0], [0], color=detail['color'], lw=2, linestyle=detail.get('linestyle', '-'), marker=detail.get('marker', None)))

    # permute labels such that it is row-major instead of column-major
    nb_cols = 4
    nb_rows = int(np.ceil(len(labels) / nb_cols))
    perm = [(i % nb_rows)*nb_cols + i // nb_rows for i in range(len(labels))]
    handles = [handles[i] for i in perm]
    labels = [labels[i] for i in perm]    

    # add legend below the axes
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0), frameon=False)

if __name__ == "__main__":
    timing_details_implicit = {
        #"copy_rhs" : {"time": time_copying_rhs_implicit, "color": 'k'},
        "solve" : {
            "time": time_solve_implicit, 
            "color": 'k', 
            'linestyle' : '-'},
        "preprocess_jac" : {
            "time": time_preprocess_jac_implicit, 
            "color": 'k', 
            'linestyle' : '--'},
        "preprocess_hess" : {
            "time": time_preprocess_hess_implicit, 
            "color": 'k', 
            'linestyle' : '-.'},
        "postprocess" : {
            "time": time_postprocess_implicit, 
            "color": 'k', 
            'linestyle' : ':'},
    }

    ### visualize flop counts
    # visualize_3d()
    # --> reformulation is much worse! we should see a nice speedup
    
    fig, axs = plt.subplots(1, 5, figsize=(13, 4))
    fig.suptitle("# FLOPs")
    show_flops_2d_plot_all_cases(ax=axs[0], independent_var='K')
    show_flops_2d_plot_all_cases(ax=axs[1], independent_var='nx')
    show_flops_2d_plot_all_cases(ax=axs[2], independent_var='nu')
    show_flops_2d_plot_all_cases(ax=axs[3], independent_var='ng')
    show_flops_2d_plot_all_cases(ax=axs[4], independent_var='ng_ineq')
    add_legend_below(fig)
    plt.savefig("unittest/ocp/figures/flop_count_scaling.png", dpi=300)
    
    ### simplified 2d figure of experimental results
    fig, axs = plt.subplots(1, 5, figsize=(13, 4))
    fig.suptitle("t_comp [us]")
    show_2d_plot_all_cases(ax=axs[0], independent_var='K', implicit_timings_details=timing_details_implicit)
    show_2d_plot_all_cases(ax=axs[1], independent_var='nx', implicit_timings_details=timing_details_implicit)
    show_2d_plot_all_cases(ax=axs[2], independent_var='nu', implicit_timings_details=timing_details_implicit)
    show_2d_plot_all_cases(ax=axs[3], independent_var='ng', implicit_timings_details=timing_details_implicit)
    show_2d_plot_all_cases(ax=axs[4], independent_var='ng_ineq', implicit_timings_details=timing_details_implicit)
    add_legend_below(fig, timing_details=timing_details_implicit)
    plt.savefig("unittest/ocp/figures/t_comp_scaling.png", dpi=300)

    plt.show()

    ### see how well flop count behaviour matches computation time
    # visualize_experiments_vs_flop_count(X_explicit, Y_explicit, Z_explicit)
    # visualize_experiments_vs_flop_count(X_reformulation, Y_reformulation, Z_reformulation)
    # visualize_experiments_vs_flop_count(X_implicit, Y_implicit, Z_implicit)

