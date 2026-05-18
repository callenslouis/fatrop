import matplotlib.pyplot as plt
import numpy as np
from plot_preparator import PlotPreparator
from helper import translate_label

def visualize_scaling_1d(df, x_func, **kwargs):
    # add default values
    y_funcs = kwargs.get('y_funcs', [])
    filters = kwargs.get('filters', [])
    y_func_linestyles = kwargs.get('y_func_linestyles', ['-', '--', '-.', ':'])
    filter_colors = kwargs.get('filter_colors', ['blue', 'black', 'red', 'green', 'orange'])
    
    pp = PlotPreparator()
    pp.set_x_metric_computer(x_func)
    if len(filters) == 0:
        filters = [pp.filter]
    if len(y_funcs) == 0:
        y_funcs = [pp.y_metric_computer]
    
    plt.figure()
        
    for f_idx, f in enumerate(filters):
        for y_idx, y in enumerate(y_funcs):
            pp.set_filter(f)
            pp.set_y_metric_computer(y)
            xx, yy_mean, yy_std = pp.prepare(df, **kwargs)
            color = filter_colors[f_idx % len(filter_colors)]
            linestyle = y_func_linestyles[y_idx % len(y_func_linestyles)]
            
            plt.plot(xx, yy_mean, linestyle=linestyle, color=color, label=f"{y.name} ({f.name})")
            plt.fill_between(xx, np.array(yy_mean) - 0.1*np.array(yy_std), np.array(yy_mean) + 0.1*np.array(yy_std), color=color, alpha=0.2)
            
    plt.axhline(0, color='k', lw=2)
        
    plt.xlabel(pp.x_metric_computer.name)
    plt.ylabel(pp.y_metric_computer.name)
    plt.title(kwargs.get('title', ''))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if kwargs.get('show', False):
        plt.show()
        
def visualize_distribution(df, metric, **kwargs):
    plt.figure()
    
    metric_vals = metric.compute_metric(df)
    plt.hist(metric_vals, bins=kwargs.get('bins', 20), color=kwargs.get('color', 'blue'), alpha=0.7)
    
    xx = np.linspace(min(metric_vals), max(metric_vals), 500)
    plt.plot(xx, np.sqrt(xx))
    
    plt.xlabel(metric.name)
    plt.ylabel('Frequency')
    plt.title(kwargs.get('title', ''))
    plt.grid()
    plt.tight_layout()
    if kwargs.get('show', False):
        plt.show()