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
    df_lu = pd.DataFrame(get_lu_data())
    
    relative_lu_difference_recursion = rel_speedup_lu_computer(recursion_benchmark_data=True).compute_metric(df)
    relative_lu_difference_lu = rel_speedup_lu_computer().compute_metric(df_lu)
    print(f"Relative LU difference (recursion benchmark data): mean={np.mean(relative_lu_difference_recursion)}, std={np.std(relative_lu_difference_recursion)}")
    print(f"Relative LU difference (LU benchmark data): mean={np.mean(relative_lu_difference_lu)}, std={np.std(relative_lu_difference_lu)}")
    
    print(f"average blocked lu time in recursion:    {np.mean(df['lu_accel']):2.4f}")
    print(f"average full lu time in recursion:       {np.mean(df['lu_reform']):2.4f}")
    print(f"average blocked lu time in lu benchmark: {np.mean(df_lu['time_blocked']):2.4f}")
    print(f"average full lu time in lu benchmark:    {np.mean(df_lu['time_full']):2.4f}")
    print(f"total blocked time in recursion:    {np.mean(df['t_accel']):2.4f}")
    print(f"total reform time in recursion:     {np.mean(df['t_reform']):2.4f}")
    # exit()
    
    
    # define filters
    areas = np.linspace(0, max(df['m']*df['n']), 7)
    filters = [size_filter(areas[i], areas[i+1]) for i in range(len(areas)-1)]
    filter_nx = [plain_df_key_filter('nx', i*5, (i+1)*5) for i in range(6)]
    filter_nu = [plain_df_key_filter('nu', i*5, (i+1)*5) for i in range(6)]
    filter_ng = [plain_df_key_filter('ng', i*5, (i+1)*5) for i in range(6)]
    filter_relevance = [lu_relevance_filter(threshold_min, threshold_max, reformulated=False) 
                        for threshold_min, threshold_max in 
                        zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:])]

    # define x-axis functions
    x_funcs = [area_computer(relative=True)]
    y_funcs_abs = [plain_df_key('t_accel'), plain_df_key('t_reform')]
    
    # define colors for filters
    first_color = [0.7, 0.7, 0.7]
    last_color = [0.0, 0.0, 0.0]
    colors = [tuple(np.array(first_color) + (np.array(last_color) - np.array(first_color)) * i / (len(filters)-1)) for i in range(len(filters))]
    
    visualize_scaling_1d(df, x_funcs[0], scatter=True, x_min_step=0.05, scatter_color_func=area_computer(relative=False),
                         file_name='relative_improvement_relative_area_scatter.png', title='Relative Improvement vs Relative Area (total area)', y_lim=(-1, 1))
    visualize_scaling_1d(df, x_funcs[0], scatter=True, x_min_step=0.05, scatter_color_func=lu_relevance_computer(reformulated=False),
                         file_name='relative_improvement_relative_area_scatter_lu_percentage.png', title='Relative Improvement vs Relative Area (lu percentage)', y_lim=(-1, 1), cbar_lims=(0, 1))
    visualize_scaling_1d(df, x_funcs[0], x_min_step=0.02, file_name='relative_improvement_relative_area.png')
    visualize_scaling_1d(df, x_funcs[0], y_funcs=[rel_speedup_lu_computer(recursion_benchmark_data=True)], x_min_step=0.02, title='Relative Speedup of LU vs Full Factorization', file_name='relative_speedup_lu_relative_area_recursion.png')
    # visualize_scaling_1d(df_lu, x_funcs[0], y_funcs=[rel_speedup_lu_computer()], x_min_step=0.02, title='Relative Speedup of LU vs Full Factorization', file_name='relative_speedup_lu_relative_area.png')
    
    for x_func in x_funcs:
        ### relative timings
        visualize_scaling_1d(df, x_func, filters=filters, x_min_step=0.05, filter_colors=colors, title='Effect of relative zero area',
                             file_name=f'relative_improvement_relative_area_binned.png')
        visualize_scaling_1d(df, x_func, filters=filter_relevance, x_min_step=0.05, filter_colors=colors, title='Effect of relative zero area',
                             file_name=f'relative_improvement_relative_area_binned_lu_relevance.png', y_lim=(-0.5, 1))
    #     visualize_scaling_1d(df, x_func, filters=filter_nx, x_min_step=0.05, filter_colors=colors, title='Effect of number of states')
    #     visualize_scaling_1d(df, x_func, filters=filter_nu, x_min_step=0.05, filter_colors=colors, title='Effect of number of controls')
    #     visualize_scaling_1d(df, x_func, filters=filter_ng, x_min_step=0.05, filter_colors=colors, title='Effect of number of constraints')
        
    #     # ### absolute timings for each
    #     # visualize_scaling_1d(df, x_func, y_funcs=y_funcs_abs, y_func_linestyles=['-', '--'], filter_colors=colors, filters=filters, x_min_step=0.1)  
    #     # visualize_scaling_1d(df, x_func, y_funcs=y_funcs_abs, y_func_linestyles=['-', '--'], filter_colors=colors, filters=filter_nx, x_min_step=0.1)  
    #     # visualize_scaling_1d(df, x_func, y_funcs=y_funcs_abs, y_func_linestyles=['-', '--'], filter_colors=colors, filters=filter_nu, x_min_step=0.1)
    #     # visualize_scaling_1d(df, x_func, y_funcs=y_funcs_abs, y_func_linestyles=['-', '--'], filter_colors=colors, filters=filter_ng, x_min_step=0.1)
        
    x_funcs = [plain_df_key('nx'), plain_df_key('nu'), plain_df_key('ng')]
    for x_func in x_funcs:
        visualize_scaling_1d(df, x_func)
        visualize_scaling_1d(df, x_func, y_funcs=[rel_speedup_lu_computer(recursion_benchmark_data=True)], title='Relative Speedup of LU vs Full Factorization', file_name=f'relative_speedup_lu_{x_func.name}_recursion.png')
        
    # visualize_scaling_1d(df, plain_df_key('nx'), y_funcs=[plain_df_key('nu'), plain_df_key('ng')])
    
    visualize_distribution(df, area_computer(relative=True))
    # visualize_distribution(df, plain_df_key('nx'))
    # visualize_distribution(df, plain_df_key('nu'))
    # visualize_distribution(df, plain_df_key('ng'))
    # visualize_distribution(df, plain_df_key('nv'))
    # visualize_distribution(df, plain_df_key('nz'))
                
    
    plt.show()
    

