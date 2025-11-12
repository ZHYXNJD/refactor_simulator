# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 19:20:13 2018

@author: kejintao

input information:
1. demand patterns (on minutes)
2. demand databases
3. drivers' working schedule (online/offline time)

** All the inputs are obtained from env, thus we do not need to alter parameters here
"""
from config import *
from path import *
import pickle
import os

class SimulatorPattern(object):
    def __init__(self,date):
        self.request_file_name = f"my_data/cleaned_orders_pickle/orders_grid{env_params['grid_num']}_{date}.pkl"
        self.driver_file_name = os.path.join(data_path, f"drivers_grid{env_params['grid_num']}_1000.pickle")
        with open(self.request_file_name, 'rb') as f:
            print(f"load request file: {self.request_file_name}")
            self.request_all = pickle.load(f)
        with open(self.driver_file_name, 'rb') as f:
            self.driver_info = pickle.load(f)
        self.driver_info = self.driver_info.sample(n=env_params['driver_num'],replace=False, random_state=42)
        # print("driver number: ",len(self.driver_info))


if __name__ == "__main__":
    simulator_pattern = SimulatorPattern('2015-05-05')
