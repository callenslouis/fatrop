from metric_computer import rel_speedup_computer
from filter import default_filter
import numpy as np
import pandas as pd

class PlotPreparator:
    def __init__(self):
        self.x_metric_computer = None
        self.y_metric_computer = rel_speedup_computer() # default
        self.filter = default_filter()
        
    def set_x_metric_computer(self, x_metric_computer):
        self.x_metric_computer = x_metric_computer
        
    def set_y_metric_computer(self, y_metric_computer):
        self.y_metric_computer = y_metric_computer
        
    def set_filter(self, filter):
        self.filter = filter

    def prepare(self, df, **kwargs):
        assert self.x_metric_computer is not None, "x_metric_computer must be set before calling prepare()"
        assert self.y_metric_computer is not None, "y_metric_computer must be set before calling prepare()"
        assert self.filter is not None, "filter must be set before calling prepare()"
        
        # get all x-values
        x_values = self.x_metric_computer.compute_unique_sorted_values(df)
        if kwargs.get('use_integer_x', False):
            x_values = [int(x) for x in x_values]
            x_values = np.unique(x_values)
        elif kwargs.get('x_min_step') is not None:
            x_min = np.min(x_values)
            x_max = np.max(x_values)
            x_min_step = kwargs.get('x_min_step')
            x_values = np.arange(x_min, x_max + x_min_step, x_min_step)
        
        # for each x value, compute the mean and std of the corresponding y values
        y_values_mean = []
        y_values_std = []
        y_all = []
        y_masks = []
        df_filtered, _ = self.filter.filter_data(df)
        # compute the x and y metrics once on the filtered data, then select the rows of each x value with a boolean mask
        x_filtered = np.asarray(self.x_metric_computer.compute_metric(df_filtered), dtype=float)
        y_filtered = self.y_metric_computer.compute_metric(df_filtered)
        binned = kwargs.get('use_integer_x') or kwargs.get('x_min_step') is not None
        for i, x in enumerate(x_values):
            if binned:
                # half-open bins [x, x_next) so that values on a bin edge are counted once; the last bin only holds x itself
                if i+1 < len(x_values):
                    mask = (x_filtered >= x) & (x_filtered < x_values[i+1])
                else:
                    mask = x_filtered == x
            else:
                mask = x_filtered == x

            y_all_i = y_filtered[mask]
            y_values_mean.append(np.mean(y_all_i))
            y_values_std.append(np.std(y_all_i))
            y_all.append(list(y_all_i))
            y_masks.append(pd.Series(mask, index=df_filtered.index))

        return x_values, y_values_mean, y_values_std, y_all, y_masks