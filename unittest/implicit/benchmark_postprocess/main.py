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
        
    df = pd.DataFrame(get_data())
    
    x_func = relative_area_computer()
    # x_func = plain_df_key('nx')
    
    visualize_scaling_1d(df, x_func)
