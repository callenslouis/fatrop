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