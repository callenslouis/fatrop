import numpy as np
from test_debug_helper import *
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import SymLogNorm
from matplotlib.colors import LinearSegmentedColormap

sys.path.append(os.path.join(os.path.dirname(__file__), "build_docker"))

from factorization_info import *
from preprocess_info import *

# preprocessing
_, _, _, _, _, _, _, _, _, _, lss_pre = get_expected_matrices(K, nu, nx, r, ng_eq, ng_ineq, modified_nu, modified_nx, modified_ng_eq, modified_ng_ineq, 
                          RSQrqt_original, GuGx_original, FuFx_original, Gg_eqt_original, Gg_ineqt_original, BAbt_original, D_x,
                          Pl, Pr, L, U, 
                          RSQrqt, GuGx, FuFx, Gg_eqt, Gg_ineqt, BAbt, Jt, store_linear_systems=True)

# recursion
blocks = GetBlockMatrices(K, modified_nu, modified_nx, modified_ng_eq, RSQrqt, GuGx, FuFx, Gg_eqt, BAbt)
solution, lss = Solve(K, modified_nu, modified_nx, modified_ng_eq, blocks['R'], blocks['S'], blocks['Q'], blocks['Gu'], blocks['Gx'], blocks['Fu'], blocks['Fx'], blocks['Hu'], blocks['Hx'], blocks['B'], blocks['A'], blocks['r'], blocks['q'], blocks['h'], blocks['b'], Pl_r, Pr_r, L_r, U_r, Lmbd, rank_k_values, Hut, -1, store_linear_systems=True)

def visualize_recursion(lss_pre, lss):
    # find the largest dimension and largest (absolute) value of the KKT matrix
    max_dim = 0
    max_value = 0
    for ls in lss:
        dim = ls['KKT'].shape[0]
        if dim > max_dim:
            max_dim = dim
        max_value = max(max_value, np.max(np.abs(ls['KKT'])))
    
    max_value = 15

    # make animation of heatmap values
    fig, (ax, ax_sol, ax_rhs) = plt.subplots(1, 3, figsize=(10, 8), 
                           gridspec_kw={'width_ratios': [max_dim, 2, 2]})
    def update(frame_idx, lss_idxs, backward_pass):
        backward_mode = backward_pass[frame_idx]
        i = lss_idxs[frame_idx]
            
        if i < 0:
            # show pre-processing
            ls = lss_pre[-(i+1)]
        else:
            # show recursion
            ls = lss[i]

        curr_dim = ls['KKT'].shape[0]
        KKT = np.block([[np.zeros((max_dim-curr_dim, max_dim))],
                        [np.zeros((curr_dim, max_dim-curr_dim)), ls['KKT']]])
        rhs = np.block([[np.zeros((max_dim-curr_dim,1))],
                        [ls['rhs']]])
        try:
            solution = np.block([[np.zeros((max_dim-curr_dim,1))],
                            [np.linalg.solve(ls['KKT'], -ls['rhs']) if backward_mode else np.zeros((curr_dim,1))]])
        except:
            # print(ls['KKT'])
            for ki in range(ls['KKT'].shape[0]):
                row = ""
                for kj in range(ls['KKT'].shape[1]):
                    elem = ls['KKT'][ki, kj]
                    if elem == 0:
                        row += "."
                    else:
                        row += "x"
                print(row)
            removed_rows = []
            for row in range(curr_dim):
                if np.all(ls['KKT'][row,:] == 0):
                    removed_rows.append(row)
            new_KKT = np.delete(ls['KKT'], removed_rows, axis=0)
            new_KKT = np.delete(new_KKT, removed_rows, axis=1)
            new_rhs = np.delete(ls['rhs'], removed_rows, axis=0)
            for ki in range(new_KKT.shape[0]):
                row = ""
                for kj in range(new_KKT.shape[1]):
                    elem = new_KKT[ki, kj]
                    if elem == 0:
                        row += "."
                    else:
                        row += "x"
                print(row)
            new_solution = np.linalg.solve(new_KKT, -new_rhs)
            for removed_rows_idx, row in enumerate(removed_rows):
                new_solution = np.insert(new_solution, row, 0)
            new_solution = np.array([[elem] for elem in new_solution])
            solution = np.block([[np.zeros((max_dim-curr_dim,1))],
                            [new_solution]])
            
        # make rhs wider by adding repeating the column
        rhs = np.hstack([rhs, rhs])
        solution = np.hstack([solution, solution])

        ax.clear(); ax_sol.clear(); ax_rhs.clear()
        norm = SymLogNorm(linthresh=1e-3, vmin=-max_value, vmax=max_value)
        cmap = 'bwr'
        cmap = LinearSegmentedColormap.from_list('custom', ['maroon', 'white', 'royalblue'])
        im = ax.imshow(KKT, cmap=cmap, norm=norm)
        ax.set_xticks([])
        ax.set_yticks([])

        ax_sol.imshow(solution, cmap=cmap, norm=norm)
        ax_sol.set_xticks([])
        ax_sol.set_yticks([])

        im_rhs = ax_rhs.imshow(rhs, cmap=cmap, norm=norm)
        ax_rhs.set_xticks([])
        ax_rhs.set_yticks([])

        if i < 0:
            fig.suptitle(f"Preprocessing" if not backward_mode else f"Postprocessing")
        else:
            fig.suptitle(f"Backward recursion" if not backward_mode else f"Forward recursion")

        return im, im_rhs

    preprocess_range = range(-1, -len(lss_pre)-1, -1) 
    pause_range = [len(lss)-1 for _ in range(7)]
    end_pause_range = [0 for _ in range(7)]
    start_pause_range = [-1 for _ in range(10)]
    recursion_range = range(len(lss)-1, -1, -1)
    lss_idxs = start_pause_range + list(preprocess_range) + pause_range + list(recursion_range) + end_pause_range
    backward_pass = [False for _ in lss_idxs] + [True for _ in lss_idxs]

    lss_idxs_flipped = lss_idxs[::-1]
    lss_idxs = lss_idxs + lss_idxs_flipped

    ani = FuncAnimation(fig, update, frames=range(len(lss_idxs)), blit=True, repeat=False, fargs=(lss_idxs, backward_pass,))
    # ani.save("recursion_visualization.gif", writer='imagemagick', fps=15)

    # save as mp4
    ani.save("recursion_visualization.mp4", writer='ffmpeg', fps=15)

visualize_recursion(lss_pre, lss)
