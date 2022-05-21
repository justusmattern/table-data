import torch
import pandas as pd
from pandas import DataFrame
import random

class TableDataset(torch.utils.data.Dataset):
    def __init__(self, data: DataFrame, column_names: list, column_type_dict: dict):
        super().__init__()

        self.data = data[column_names]
        self.column_type_dict = column_type_dict
        self.column_names = column_names

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        values = [(self.data[name].to_list()[index], self.column_type_dict[name], name) for name in self.column_names]
        values = random.sample(values, len(values))

        target_string = self.values_to_string(values)
        numbers = [val for val, c_type, c_name in values if c_type == 'num']
        #print(target_string)
        #print(numbers)
        return target_string, numbers


    def values_to_string(self, values):
        substrings = []
        for value, val_type, col_name in values:
            if val_type == 'num':
                col = f'{col_name}: [NUM]'
            else: 
                col = f'{col_name}: {value}'
            substrings.append(col)
        
        full_string = " ||| ".join(substrings)
        return full_string
