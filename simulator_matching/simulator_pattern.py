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
    def __init__(self,date=None):
        # read parameters
        # self.request_file_name = kwargs['request_file_name']
        # self.driver_file_name = kwargs['driver_file_name']
        # orders_grid35_2015-05-04.pickle
        # drivers_grid35_1000.pickle
        if date is None:
            self.request_file_name = os.path.join(data_path, f"orders_grid{env_params['grid_num']}_{env_params['date']}.pickle")
            print(f"load default date from docker: {env_params['date']}")
        else:
            self.request_file_name = f"my_data/orders_grid{env_params['grid_num']}_{date}.pkl"
            print(f"load date from my disk: {date}")
        self.driver_file_name = os.path.join(data_path, f"drivers_grid{env_params['grid_num']}_1000.pickle")
        with open(self.request_file_name, 'rb') as f:
            print(f"load docker request file: {self.request_file_name}")
            self.request_all = pickle.load(f)
        # with open(f"my_data/orders_grid{env_params['grid_num']}_2015-05-04.pkl", 'rb') as f:
        #     print(f"load my request file:my_data/orders_grid{env_params['grid_num']}_2015-05-04.pkl")
        #     self.request_mydt = pickle.load(f)
        with open(self.driver_file_name, 'rb') as f:
            self.driver_info = pickle.load(f)
        self.driver_info = self.driver_info.sample(n=env_params['driver_num'],replace=False, random_state=42)
        # print("driver number: ",len(self.driver_info))


if __name__ == "__main__":
    simulator_pattern = SimulatorPattern()
