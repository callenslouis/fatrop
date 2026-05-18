import matplotlib.pyplot as plt
from plot_preparator import PlotPreparator
from helper import translate_label

def visualize_scaling_1d(df, x_func, **kwargs):
    # add default values
    y_funcs = kwargs.get('y_funcs', [])
    filters = kwargs.get('filters', [])
    
    pp = PlotPreparator()
    pp.set_x_metric_computer(x_func)
    if len(filters) == 0:
        filters = [pp.filter]
    if len(y_funcs) == 0:
        y_funcs = [pp.y_metric_computer]
    
    plt.figure()
        
    for f in filters:
        for y in y_funcs:
            pp.set_filter(f)
            pp.set_y_metric_computer(y)
            xx, yy_mean, yy_std = pp.prepare(df, **kwargs)
            plt.errorbar(xx, yy_mean, yerr=yy_std, fmt='-o', label=f"{y.name} ({f.name})")
        
    plt.xlabel(pp.x_metric_computer.name)
    plt.ylabel(pp.y_metric_computer.name)
    plt.title(kwargs.get('title', ''))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if kwargs.get('show', False):
        plt.show()