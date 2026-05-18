import numpy as np

# abstract base class for metric computer
# for all data entries, it computes the x-metric value
class metric_computer:
    def __init__(self, name):
        self.name = name
        
    # Helper function to compute the metric
    def compute_metric(self, df):
        # compute the metric for all rows and add it as a new column to the dataframe
        raise NotImplementedError
    
    def filter_df(self, df, x_value):
        # filter the dataframe to only include rows where the metric value is equal to x_value
        return df[self.compute_metric(df) == x_value]

    # compute all values of the metric
    def compute_unique_sorted_values(self, df):
        return np.sort(np.unique(self.compute_metric(df)))
    
    # evaluate the metric on the given dataframe, returning mean and std
    def evaluate(self, df_filtered):
        m = self.compute_metric(df_filtered)
        return np.mean(m), np.std(m)
    
    
# specific computer
class plain_df_key(metric_computer):
    def __init__(self, key):
        super().__init__(key)
        self.key = key
        
    def compute_metric(self, df):
        return df[self.key]
    
class relative_area_computer(metric_computer):
    def __init__(self):
        super().__init__('Relative Area')
        
    def compute_metric(self, df):
        return df['m_rel'] * df['n_rel']
    
class rel_speedup_computer(metric_computer):
    def __init__(self):
        super().__init__('Relative Speedup')
        
    def compute_metric(self, df):
        return (df['t_accel'] - df['t_reform']) / df['t_reform']
