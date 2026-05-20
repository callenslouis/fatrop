import numpy as np

# abstract base class for metric computer
# for all data entries, it computes the x-metric value
class metric_computer:
    def __init__(self, name):
        self.name = name
        
    # Helper function to compute the metric
    def compute_metric(self, df):
        # compute the metric for all rows
        raise NotImplementedError
    
    def filter_df(self, df, x_value):
        # filter the dataframe to only include rows where the metric value is equal to x_value
        mask = self.compute_metric(df) == x_value
        return df[mask], mask
    
    def filter_df_range(self, df, x_min, x_max):
        # filter the dataframe to only include rows where the metric value is between x_min and x_max
        mask = (self.compute_metric(df) >= x_min) & (self.compute_metric(df) <= x_max)
        return df[mask], mask

    # compute all values of the metric
    def compute_unique_sorted_values(self, df):
        m = self.compute_metric(df)
        m = m[~np.isnan(m)]
        return np.sort(np.unique(m))
    
    # evaluate the metric on the given dataframe, returning mean and std
    def evaluate(self, df_filtered):
        m = self.compute_metric(df_filtered)
        return np.mean(m), np.std(m), m
    
    
# specific computer
class plain_df_key(metric_computer):
    def __init__(self, key):
        super().__init__(key)
        self.key = key
        
    def compute_metric(self, df):
        return df[self.key]
    
class area_computer(metric_computer):
    def __init__(self, relative=False):
        super().__init__('Relative area of zero block')
        self.relative = relative
        
    def compute_metric(self, df):
        if self.relative:
            return df['m_rel'] * df['n_rel']
        
        return df['m'] * df['n']
    
class square_root_area_computer(metric_computer):
    def __init__(self):
        super().__init__(f'$\sqrt{{mn}}$')
        
    def compute_metric(self, df):
        return np.sqrt(df['m'] * df['n'])

class rel_speedup_computer(metric_computer):
    def __init__(self):
        super().__init__('Relative Speedup')
        
    def compute_metric(self, df):
        return (df['t_accel'] - df['t_reform']) / df['t_reform']
    
class rel_speedup_lu_computer(metric_computer):
    def __init__(self, recursion_benchmark_data=False):
        super().__init__('Relative Speedup LU')
        self.recursion_benchmark_data = recursion_benchmark_data

    def compute_metric(self, df):
        if self.recursion_benchmark_data:
            return (df['lu_accel'] - df['lu_reform']) / df['lu_reform']
        else:
            return (df['time_blocked'] - df['time_full']) / df['time_full']

class lu_relevance_computer(metric_computer):
    def __init__(self, reformulated=False):
        super().__init__('Percentage of recursion time spent in LU decomposition')
        self.reformulated = reformulated

    def compute_metric(self, df):
        if self.reformulated:
            return df['lu_reform'] / df['t_reform']
        else:
            return df['lu_accel'] / df['t_accel']
        