import numpy as np

class Filter:
    def __init__(self, name):
        self.name = name       
        
    # returns true or false for each row in the dataframe, indicating whether the row should be included in the plot
    def filter_pass_condition(self, df):
        raise NotImplementedError
    
    def filter_data(self, df):
        return df[self.filter_pass_condition(df)]
    
    
class default_filter(Filter):
    def __init__(self):
        super().__init__('Default Filter')
        
    def filter_pass_condition(self, df):
        # return true for all rows of df
        return np.ones(len(df), dtype=bool)
    
class size_filter(Filter):
    def __init__(self, area_min, area_max):
        super().__init__(f'{area_min:.2f} $<$ area $<$ {area_max:.2f}')
        self.area_min = area_min
        self.area_max = area_max
        
    def filter_pass_condition(self, df):
        area = df['m'] * df['n']
        return (area >= self.area_min) & (area <= self.area_max)