# Written by Brian Day

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

results_filepath = '/Users/brian_day/Desktop/WilmerLab_Research/Research_Projects/Gas_Sensing/CO2_RASPA_Results/CO2_1bar/CO2_Sensing_Results_New_New/exp_Data_files/'
all_mof_results_filename = '/Users/brian_day/Desktop/WilmerLab_Research/Research_Projects/Gas_Sensing/CO2_RASPA_Results/CO2_1bar/CO2_Sensing_Results_New_New/ALL_MOFS.csv'

col_names = np.array(['Run_ID','MOF','Mass','CO2','O2','N2'])

filename = all_mof_results_filename
all_ads_results = pd.read_csv(filename, delimiter='\t|,|:|;', engine='python', header=None, names=col_names)
all_ads_results = all_ads_results.drop(index=0)
all_ads_results = all_ads_results.values

all_run_ids = np.linspace(0,960,961)
for id in all_run_ids:
    exp_run_id = int(id)
    exp_data = np.array(['Run_ID','MOF','Mass','CO2','O2','N2'])
    for i in range(len(all_ads_results[:,0])):
        if int(all_ads_results[i,0]) == exp_run_id:
            exp_data = np.vstack([exp_data, all_ads_results[i,:]])

    exp_data_filename = results_filepath+'exp_data_'+str(exp_run_id)+'.csv'
    with open(exp_data_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(exp_data)
