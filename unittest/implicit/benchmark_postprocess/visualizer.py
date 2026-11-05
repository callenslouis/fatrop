import matplotlib.pyplot as plt
import numpy as np
from plot_preparator import PlotPreparator
from helper import translate_label
import yaml
settings = yaml.safe_load(open('settings.yaml', 'r'))

def visualize_scaling_1d(df, x_func, **kwargs):
    print(f"Started visualize_scaling_1d")
    pp = PlotPreparator()
    pp.set_x_metric_computer(x_func)
 
    # add default values
    y_funcs = kwargs.get('y_funcs', [pp.y_metric_computer])
    filters = kwargs.get('filters', [pp.filter])
    y_func_linestyles = kwargs.get('y_func_linestyles', ['-', '--', '-.', ':'])
    filter_colors = kwargs.get('filter_colors', ['black', 'blue', 'red', 'green', 'orange'])
    
    plt.figure()
    
    for f_idx, f in enumerate(filters):
        for y_idx, y in enumerate(y_funcs):
            print(f"Processing filter {f.name} and y_func {y.name}")
            pp.set_filter(f)
            pp.set_y_metric_computer(y)
            # the unbinned (fine) data is only needed for the scatter plot
            if kwargs.get('scatter', False):
                kwargs_fine = kwargs.copy()
                kwargs_fine['x_min_step'] = None
                kwargs_fine['use_integer_x'] = False
                xx_fine, yy_mean_fine, yy_std_fine, y_all_fine, y_masks_fine = pp.prepare(df, **kwargs_fine)
            xx, yy_mean, yy_std, y_all, y_masks = pp.prepare(df, **kwargs)
            color = filter_colors[f_idx % len(filter_colors)]
            linestyle = y_func_linestyles[y_idx % len(y_func_linestyles)]

            if len(y_funcs) > 1 and len(filters) > 1:
                label = f"{y.name} ({f.name})"
            elif len(y_funcs) > 1:
                label = f"{y.name}"
            elif len(filters) > 1:
                label = f"{f.name}"
            else:
                label = None

            plt.plot(xx, yy_mean, linestyle=linestyle, color=color, label=f"{label}")
            if kwargs.get('scatter', False):
                y_all_flat = []
                xx_flat = []                
                colors = []
                for i in range(len(xx_fine)):
                    xx_flat.extend([xx_fine[i]] * len(y_all_fine[i]))
                    y_all_flat.extend(y_all_fine[i])
                    if kwargs.get('scatter_color_func') is not None:
                        df_entries = df[y_masks_fine[i]]
                        color_for_entries = kwargs.get('scatter_color_func').compute_metric(df_entries)
                        colors.extend(color_for_entries)
                    else:
                        colors.extend([color] * len(y_all_fine[i]))
                
                sc = plt.scatter(xx_flat, y_all_flat, c=colors, s=0.1, linestyle=linestyle, label=label, alpha=1.0)
                plt.colorbar(sc, label=kwargs.get('scatter_color_func').name)
                sc.set_cmap('jet')
            elif kwargs.get('show_std', True):
                std_scale = kwargs.get('std_scale', 1.0)
                plt.fill_between(xx, np.array(yy_mean) - std_scale*np.array(yy_std), np.array(yy_mean) + std_scale*np.array(yy_std), color=color, alpha=0.2)
            
    plt.axhline(0, color='k', lw=2)

    plt.xlim(min(xx), max(xx))
    if kwargs.get('y_lim', None) is not None:
        print(f"Setting y limits to {kwargs.get('y_lim')}")
        plt.ylim(kwargs.get('y_lim'))
    if kwargs.get('cbar_lims', None) is not None:
        print(f"Setting colorbar limits to {kwargs.get('cbar_lims')}")
        plt.clim(kwargs.get('cbar_lims'))
    plt.xlabel(pp.x_metric_computer.name)
    plt.ylabel(pp.y_metric_computer.name)
    plt.title(kwargs.get('title', ''))
    plt.legend(ncol=2)
    plt.grid()
    plt.tight_layout()
    
    if kwargs.get('file_name', None) is not None:
        print(f"Saving figure to {kwargs.get('file_name')}")
        plt.savefig(settings.get('figure_folder', 'figures-thesis') + '/' + kwargs.get('file_name'), dpi=300)
    
    if kwargs.get('show', False):
        plt.show()
        
def visualize_boxplot(df, y_funcs, filters, **kwargs):
    # one group of boxes per filter (e.g. a bin of the matrix size), with one box per y_func showing its distribution
    # over the problems in that bin
    if not isinstance(y_funcs, (list, tuple)):
        y_funcs = [y_funcs]
    colors = kwargs.get('colors', [kwargs.get('color', 'grey')] if len(y_funcs) == 1 else ['lightgrey', 'dimgrey'])
    group_width = 0.6
    box_width = group_width / len(y_funcs)

    plt.figure()

    tick_labels = []
    legend_handles = []
    for y_idx, y_func in enumerate(y_funcs):
        data = []
        for f in filters:
            df_filtered, _ = f.filter_data(df)
            values = np.asarray(y_func.compute_metric(df_filtered), dtype=float)
            data.append(values[~np.isnan(values)])
            if y_idx == 0:
                tick_labels.append(f"{f.value_min:g}--{f.value_max:g}\n($n = {len(df_filtered)}$)")
        positions = np.arange(1, len(filters) + 1) - group_width / 2 + (y_idx + 0.5) * box_width
        # whiskers at the 5th and 95th percentile, outliers are not drawn
        bp = plt.boxplot(data, positions=positions, widths=0.9*box_width, whis=kwargs.get('whis', (5, 95)),
                         showfliers=kwargs.get('show_fliers', False), patch_artist=True, manage_ticks=False,
                         boxprops=dict(facecolor=colors[y_idx % len(colors)], alpha=0.7), medianprops=dict(color='k', lw=2))
        legend_handles.append(bp['boxes'][0])
    plt.xticks(range(1, len(tick_labels) + 1), tick_labels)
    plt.xlim(0.5, len(tick_labels) + 0.5)
    plt.axhline(0, color='k', lw=1)

    if len(y_funcs) > 1:
        plt.legend(legend_handles, kwargs.get('labels', [y.name for y in y_funcs]))
    if kwargs.get('y_lim', None) is not None:
        plt.ylim(kwargs.get('y_lim'))
    plt.xlabel(kwargs.get('x_label', filters[0].metric_computer.name))
    plt.ylabel(kwargs.get('y_label', y_funcs[0].name))
    plt.title(kwargs.get('title', ''))
    plt.grid(axis='y')
    plt.tight_layout()

    if kwargs.get('file_name', None) is not None:
        print(f"Saving figure to {kwargs.get('file_name')}")
        plt.savefig(settings.get('figure_folder', 'figures-thesis') + '/' + kwargs.get('file_name'), dpi=300)

    if kwargs.get('show', False):
        plt.show()

def visualize_distribution(df, metric, **kwargs):
    plt.figure()
    
    metric_vals = metric.compute_metric(df)
    
    plt.hist(metric_vals, bins=kwargs.get('bins', 20), color=kwargs.get('color', 'blue'), alpha=0.7, density=kwargs.get('normalized', False))
    
    if kwargs.get('show_median', False):
        median = np.median(metric_vals)
        plt.axvline(median, color='k', linestyle='-', linewidth=2)
        # plt.text(median, plt.ylim()[1]*0.1, f'Median: {median:.2f}', color='k', ha='right', fontsize=10)
    
    if kwargs.get('x_lim', None) is not None:
        plt.xlim(kwargs.get('x_lim'))
    
    plt.xlabel(metric.name)
    # with density=True the bar heights are a probability density (the histogram integrates to 1), not counts
    plt.ylabel('Probability density' if kwargs.get('normalized', False) else 'Number of problems')
    plt.title(kwargs.get('title', ''))
    plt.grid()
    plt.tight_layout()
    
    if kwargs.get('file_name', None) is not None:
        print(f"Saving figure to {kwargs.get('file_name')}")
        plt.savefig(settings.get('figure_folder', 'figures-thesis') + '/' + kwargs.get('file_name'), dpi=300)
    
    if kwargs.get('show', False):
        plt.show()