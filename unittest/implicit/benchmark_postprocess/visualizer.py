import matplotlib.pyplot as plt
from plot_preparator import PlotPreparator
from helper import translate_label

def visualize_scaling_1d(df, x_func, y_func=None, filters=[], **kwargs):
    pp = PlotPreparator()
    pp.set_x_metric_computer(x_func)
    if y_func is not None:
        pp.set_y_metric_computer(y_func)
    
    plt.figure()
    if len(filters) == 0:
        filters = [pp.filter]
        
    for f in filters:
        pp.set_filter(f)
        xx, yy_mean, yy_std = pp.prepare(df)
        plt.errorbar(xx, yy_mean, yerr=yy_std, fmt='-o', label=f.name)
        
    # plt.xlabel(translate_label(x_func.name))
    # plt.ylabel(translate_label(y_func.name))
    plt.title(kwargs.get('title', ''))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()