from metric_computer import rel_speedup_computer
from filter import default_filter

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

    def prepare(self, df):
        assert self.x_metric_computer is not None, "x_metric_computer must be set before calling prepare()"
        assert self.y_metric_computer is not None, "y_metric_computer must be set before calling prepare()"
        assert self.filter is not None, "filter must be set before calling prepare()"
        
        # get all x-values
        x_values = self.x_metric_computer.compute_unique_sorted_values(df)
        
        # for each x value, compute the mean and std of the corresponding y values
        y_values_mean = []
        y_values_std = []
        for x in x_values:
            y_mean, y_std = self.y_metric_computer.evaluate(self.x_metric_computer.filter_df(self.filter.filter_data(df), x))
            y_values_mean.append(y_mean)
            y_values_std.append(y_std)

        return x_values, y_values_mean, y_values_std