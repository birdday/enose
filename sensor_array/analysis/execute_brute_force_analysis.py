#!/usr/bin/env python

# --------------------------------------------------
# ----- Import Python Packages ---------------------
# --------------------------------------------------
import sys
import pandas as pd
from datetime import datetime
from brute_force_analysis import (
    read_data_as_dict,
    write_data_as_tabcsv,
    yaml_loader,
    import_experimental_data,
    import_simulated_data,
    create_comp_list,
    moving_average_smooth,
    reintroduce_random_error,
    convert_experimental_data,
    calculate_element_pmf,
    calculate_all_arrays,
    create_bins,
    bin_compositions,
    calculate_kld,
    choose_arrays,
    assign_array_ids,
    save_element_pmf_data,
    save_unbinned_array_pmf_data,
    plot_element_mass_data,
    plot_unbinned_array_pmf_data,
    save_binned_array_pmf_data,
    plot_binned_array_pmf_data)

# --------------------------------------------------
# ----- Import RASPA Data and yaml File ------------
# --------------------------------------------------
# Import yaml file as dictoncary
filepath = 'config_files/process_config.sample.yaml'
data = yaml_loader(filepath)

# Redefine key variables in yaml file
sim_data = data['sim_data']
exp_data = data['exp_data']
num_mofs = data['number_mofs']
num_mixtures = data['num_mixtures']
num_bins = data['num_bins']
num_best_worst = data['num_best_worst']
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
exp_results_import = read_data_as_dict(exp_data)

# --------------------------------------------------
# ----- Calculate arrays, PMFs, KLDs, etc. ---------
# --------------------------------------------------
exp_results_full, exp_results_mass, exp_mof_list = \
    import_experimental_data(exp_results_import, mof_list, mof_densities, gases)
sim_results_full = \
    import_simulated_data(sim_results_import, mof_list, mof_densities, gases)
comp_list, mole_fractions = \
    create_comp_list(sim_results_full, mof_list, gases)
sim_results_full = \
    moving_average_smooth(sim_results_full, mof_list, gases, comp_list, mole_fractions, num_points = 2)
sim_results_full = \
    reintroduce_random_error(sim_results_full, error=1, seed=0)
exp_results_full = \
    convert_experimental_data(exp_results_full, sim_results_full, mof_list, gases)
element_pmf_results = \
    calculate_element_pmf(exp_results_full, sim_results_full, mof_list, stdev, mrange)
list_of_arrays, all_array_pmf_results = \
    calculate_all_arrays(mof_list, num_mofs, element_pmf_results, gases)
bins = \
    create_bins(gases, num_bins, mof_list, element_pmf_results)
binned_probabilities_sum, binned_probabilities_max = \
    bin_compositions(gases, bins, list_of_arrays, all_array_pmf_results)
array_kld_results = \
    calculate_kld(gases, list_of_arrays, bins, all_array_pmf_results, binned_probabilities_sum)
best_and_worst_arrays_by_absKLD, best_and_worst_arrays_by_jointKLD, best_and_worst_arrays_by_gasKLD = \
    choose_arrays(gases, num_mofs, array_kld_results, num_best_worst)

# --------------------------------------------------
# ----- Choose what to save ------------------------
# --------------------------------------------------
# element_pmf_results_df = pd.DataFrame(data=element_pmf_results)
list_of_array_ids = assign_array_ids(list_of_arrays)
timestamp = (datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
save_element_pmf_data(element_pmf_results_df, stdev, mrange, timestamp)
save_unbinned_array_pmf_data(gases, list_of_arrays, list_of_array_ids, all_array_pmf_results, timestamp)
plot_unbinned_array_pmf_data(gases, list_of_arrays, list_of_array_ids, all_array_pmf_results, timestamp)
save_binned_array_pmf_data(gases, list_of_arrays, list_of_array_ids, bins, binned_probabilities_sum, timestamp)
plot_binned_array_pmf_data(gases, list_of_arrays, list_of_array_ids, bins, binned_probabilities_sum, timestamp)
write_data_as_tabcsv('saved_array_kld/best_and_worst_arrays_by_absKLD_%s.csv' % timestamp, best_and_worst_arrays_by_absKLD)
write_data_as_tabcsv('saved_array_kld/best_and_worst_arrays_by_jointKLD_%s.csv' % timestamp, best_and_worst_arrays_by_jointKLD)
write_data_as_tabcsv('saved_array_kld/best_and_worst_arrays_by_gasKLD_%s.csv' % timestamp, best_and_worst_arrays_by_gasKLD)
