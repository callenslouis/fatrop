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
    areas = np.linspace(0, max(df['m']*df['n']), 8)
    filters = [size_filter(areas[i], areas[i+1]) for i in range(len(areas)-1)]
    
    # define x-axis functions
    x_funcs = [area_computer(relative=True)]#, plain_df_key('m_rel'), plain_df_key('n_rel')]
    
    ### relative timings
    for x_func in x_funcs:
        visualize_scaling_1d(df, x_func, filters=filters, x_min_step=0.1)
    plt.show()
    
    ### absolute timings for each
    # y_funcs = [plain_df_key('t_accel'), plain_df_key('t_reform')] #, plain_df_key('t_impl')]
    # for x_func in x_funcs:
    #     visualize_scaling_1d(df, x_func, y_funcs=y_funcs, filters=filters, x_min_step=0.1)
    visualize_scaling_1d(df, x_func, y_funcs=[plain_df_key('t_accel'), plain_df_key('t_reform')], 
                         y_func_linestyles=['-', '--'], filters=filters, 
                         x_min_step=0.1)
    # plt.show()
    
    visualize_distribution(df, area_computer(relative=False), show=True)

