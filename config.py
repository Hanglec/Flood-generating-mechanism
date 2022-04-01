import os
import sys

import albumentations as A
from types import SimpleNamespace
import argparse
import numpy as np

abs_path = os.path.dirname(__file__)  # 获取当前运行脚本的绝对路径
args = {
    'data_path': r'I:\RE03\RE03_Results',
    'pre_idmax_path': r'\pre_idmax1979-2013.npy',
    'soil_idmax_path': r'\soil_idmax1979-2013.npy',
    'snow_idmax_path': r'\snow_idmax1979-2013.npy',
    'runoff_idmax_path': r'\runoff_idmax1979-2013.npy',

    'results_path': r'I:\RE03\Results_20220324',
    'regions_path': r'\regions.npy',
    'Alpha_path': r'\results_Alpha.npy',
    'Beta_path': r'\results_Beta.npy',
    'set_path': r'\csv_set.csv',
    'RFresults': r'\csv_RF_results692.csv',
    'RFresults8': r'\csv_RF8_461.csv',
    'matrix3': r'\matrix_class3.npy',
    'matrix8': r'\matrix_class8.npy',
    'cluster_score': r'\kcluster_score.npy',

    'study_years': range(1979, 2014),
    'global_lon': 1440,
    'lon_array': np.linspace(-179.875, 179.875, 1440),
    'global_lat': 600,
    'lat_array': np.linspace(-59.875, 89.875, 600),
    'valid_years': 20,

    'fonts': 26,
    'seed': 2022,
    'RF_split': 10,
    'grids': 234522




}


parser = argparse.ArgumentParser(description='')
parser.add_argument('-C', '--config', help='config filename', default=args)
parser_args, _ = parser.parse_known_args(sys.argv)
args = SimpleNamespace(**args)
print('Using config file', parser_args.config)