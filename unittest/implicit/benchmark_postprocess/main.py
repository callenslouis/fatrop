import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from helper import *
from metric_computer import *
from plot_preparator import *
from filter import *
from visualizer import *
settings = json.load(open('../post_process_settings.json', 'r'))


if __name__ == '__main__':
    if settings.get('latexify', True):
        latexify()
        
    # load data
    df = pd.DataFrame(get_data())
    
    # define filters
    areas = np.linspace(0, max(df['m']*df['n']), 7)
    filters = [size_filter(areas[i], areas[i+1]) for i in range(len(areas)-1)]
    filter_nx = [plain_df_key_filter('nx', i*5, (i+1)*5) for i in range(6)]
    filter_nu = [plain_df_key_filter('nu', i*5, (i+1)*5) for i in range(6)]
    filter_ng = [plain_df_key_filter('ng', i*5, (i+1)*5) for i in range(6)]

    # define x-axis functions
    x_funcs = [area_computer(relative=True)]
    y_funcs_abs = [plain_df_key('t_accel'), plain_df_key('t_reform')]
    
    # define colors for filters
    first_color = [0.7, 0.7, 0.7]
    last_color = [0.0, 0.0, 0.0]
    colors = [tuple(np.array(first_color) + (np.array(last_color) - np.array(first_color)) * i / (len(filters)-1)) for i in range(len(filters))]
    
    visualize_scaling_1d(df, x_funcs[0], scatter=True, x_min_step=0.05, scatter_color_func=area_computer(relative=False))
    
    
    for x_func in x_funcs:
        ### relative timings
        visualize_scaling_1d(df, x_func, filters=filters, x_min_step=0.05, filter_colors=colors, title='Effect of relative zero area')
        visualize_scaling_1d(df, x_func, filters=filter_nx, x_min_step=0.05, filter_colors=colors, title='Effect of number of states')
        visualize_scaling_1d(df, x_func, filters=filter_nu, x_min_step=0.05, filter_colors=colors, title='Effect of number of controls')
        visualize_scaling_1d(df, x_func, filters=filter_ng, x_min_step=0.05, filter_colors=colors, title='Effect of number of constraints')
        
        # ### absolute timings for each
        # visualize_scaling_1d(df, x_func, y_funcs=y_funcs_abs, y_func_linestyles=['-', '--'], filter_colors=colors, filters=filters, x_min_step=0.1)  
        # visualize_scaling_1d(df, x_func, y_funcs=y_funcs_abs, y_func_linestyles=['-', '--'], filter_colors=colors, filters=filter_nx, x_min_step=0.1)  
        # visualize_scaling_1d(df, x_func, y_funcs=y_funcs_abs, y_func_linestyles=['-', '--'], filter_colors=colors, filters=filter_nu, x_min_step=0.1)
        # visualize_scaling_1d(df, x_func, y_funcs=y_funcs_abs, y_func_linestyles=['-', '--'], filter_colors=colors, filters=filter_ng, x_min_step=0.1)
        
    x_funcs = [plain_df_key('nx'), plain_df_key('nu'), plain_df_key('ng')]
    for x_func in x_funcs:
        visualize_scaling_1d(df, x_func)
        
    # visualize_scaling_1d(df, plain_df_key('nx'), y_funcs=[plain_df_key('nu'), plain_df_key('ng')])
    
    # visualize_distribution(df, plain_df_key('nx'))
    # visualize_distribution(df, plain_df_key('nu'))
    # visualize_distribution(df, plain_df_key('ng'))
    # visualize_distribution(df, plain_df_key('nv'))
    # visualize_distribution(df, plain_df_key('nz'))
                
    # df = pd.DataFrame(get_lu_data())
    # y_funcs = [rel_speedup_lu_computer()]
    # visualize_scaling_1d(df, x_funcs[0], y_funcs=y_funcs, filters=filters, x_min_step=0.1, title='Relative Speedup of LU vs Full Factorization')
    
    plt.show()
    

