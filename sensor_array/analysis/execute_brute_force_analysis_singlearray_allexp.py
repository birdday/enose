#!/usr/bin/env python

# --------------------------------------------------
# ----- Import Python Packages ---------------------
# --------------------------------------------------
import csv
import sys
import pandas as pd
from datetime import datetime
from brute_force_analysis import (
    read_data_as_dict,
    write_data_as_tabcsv,
    yaml_loader,
    import_experimental_data,
    import_simulated_data,
    calculate_element_pmf,
    calculate_array_pmf,
    calculate_all_arrays_list,
    calculate_all_arrays,
    calculate_single_array_kld,
    create_bins,
    create_comp_set_dict,
    bin_compositions,
    bin_compositions_single_array,
    calculate_kld,
    choose_arrays,
    assign_array_ids,
    save_element_pmf_data,
    save_unbinned_array_pmf_data,
    plot_element_mass_data,
    plot_unbinned_array_pmf_data,
    save_binned_array_pmf_data,
    plot_binned_array_pmf_data)
from genetic_algorithm_analysis import read_GA_results_messy

# --------------------------------------------------
# ----- Import RASPA Data and yaml File ------------
# --------------------------------------------------
# Import yaml file as dictoncary
filepath = 'config_files/process_config.sample_allkld.yaml'
data = yaml_loader(filepath)

# Redefine key variables in yaml file
sim_data = data['sim_data']
num_mofs = data['number_mofs']
num_bins = data['num_bins']
stdev = data['stdev']
mrange = data['mrange']
gases = data['gases']
mof_list = data['mof_list']
mof_densities = {}
for mof in mof_list:
    mof_densities.copy()
    mof_densities.update({ mof : data['mofs'][mof]['density']})

# Import results as dictionary
sim_results_import = read_data_as_dict(sim_data)
sim_results_full = \
    import_simulated_data(sim_results_import, mof_list, mof_densities, gases)

# --------------------------------------------------
# ----- Calculate arrays, PMFs, KLDs, etc. ---------
# --------------------------------------------------
array = ['ALUKIC']

import numpy as np
# list_of_experiments = np.linspace(1,961,961)
list_of_experiments = np.linspace(0,960,9611)

all_kld_results = []
for exp in list_of_experiments:
    # Define Filepath
    exp_data = data['exp_data_path']+'exp_data_'+str(int(exp))+'.csv'
    exp_results_import = read_data_as_dict(exp_data)

    # Import Corresponding Results
    exp_results_full, exp_results_mass, exp_mof_list = \
        import_experimental_data(exp_results_import, mof_list, mof_densities, gases)
    element_pmf_results = \
        calculate_element_pmf(exp_results_full, sim_results_full, mof_list, stdev, mrange)
    bins = \
         create_bins(gases, num_bins, mof_list, element_pmf_results)
    comp_set_dict = \
        create_comp_set_dict(element_pmf_results, mof_list)

    # Round Bins (Fix Later)
    import numpy as np
    for row in bins:
        for gas in gases:
            row[gas] = np.round(row[gas],4)

    # Calculate KLD
    single_array_pmf_results = \
        calculate_array_pmf(array, element_pmf_results)
    binned_probabilities_sum, array_dict = \
        bin_compositions_single_array(gases, bins, array, single_array_pmf_results, comp_set_dict)
    array_kld_results = \
        calculate_single_array_kld(gases, array, bins, single_array_pmf_results, binned_probabilities_sum)
    array_kld_results['run_id'] = int(exp)
    all_kld_results.append(array_kld_results)
    
