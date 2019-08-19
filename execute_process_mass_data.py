#!/usr/bin/env python

# --------------------------------------------------
# ----- Import Python Packages ---------------------
# --------------------------------------------------
import sys
import pandas as pd
from datetime import datetime
from mof_array.pmf.process_mass_data import *
# List of Functions and Arguments in process_mass_data:
#     read_data_as_dict(filename)
#           --> return list(output_data)
#     write_data_as_tabcsv(filename, data)
#           --> return(writer)
#     yaml_loader(filepath)
#           --> return(data)
#     import_experimental_data(exp_results_import, mof_list, mof_densities, gases)
#           --> return(exp_results_full, exp_results_mass, exp_mof_list)
#     import_simulated_data(sim_results_import, mof_list, mof_densities, gases)
#           --> return(sim_results_full)
#     add_random_gas(gases, comps, num_mixtures)
#           --> NOT FUCTIONAL CURRENTLY
#     calculate_pmf(exp_results_full, sim_results_full, mof_list, stdev, mrange)
#           --> return(element_pmf_results)
#     caclulate_array_pmf(mof_array, element_pmf_results)
#           --> return(single_array_pmf_results)
#     calculate_all_arrays(mof_list, num_mofs, element_pmf_results, gases)
#           --> return(mof_array_list, all_array_pmf_results)
#     create_bins(gases, num_bins, mof_list, element_pmf_results)
#           --> return(bins)
#     bin_compositions(gases, bins, list_of_arrays, all_array_pmf_results)
#           --> return(binned_probabilities_sum, binned_probabilities_max)
#     plot_binned_pmf_array(gases, list_of_arrays, bins, binned_probabilities)
#           --> returns a set of plots to the folder 'figures'
#     save_array_pmf_data(gases, list_of_arrays, bins, binned_probabilities)
#           --> returns a set of csv files to the folder 'saved_array_pmfs'
#     save_element_pmf_data(element_pmf_results, stdev, mrange, num_mofs)
#           --> returns a set of csv files to the folder 'saved_element_pmfs'
#     calculate_kld(gases, list_of_arrays, bins, binned_probabilities)
#           --> return(array_kld_results)
#     choose_arrays(gases, num_mofs, array_kld_results, num_best_worst)
#           --> return(best_and_worst_arrays_by_jointKLD,
#               best_and_worst_arrays_by_gasKLD, best_ranked_by_product)

# --------------------------------------------------
# ----- Import RASPA Data and yaml File ------------
# --------------------------------------------------
# Redefine system arguments
# sim_data = sys.argv[1]
# exp_data = sys.argv[2]
sim_data = 'ALL_MOFS.csv'
exp_data = 'exp_data_175.csv'

# Import results as dictionary
sim_results_import = read_data_as_dict(sim_data)
exp_results_import = read_data_as_dict(exp_data)

# Import yaml file as dictoncary
filepath = 'settings/process_config.yaml'
data = yaml_loader(filepath)

# Redefine key varaibles in yaml file
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

# --------------------------------------------------
# ----- Calculate arrays, PMFs, KLDs, etc. ---------
# --------------------------------------------------
# N.B. Parentheses are only necessary for implicit line continuation
exp_results_full, exp_results_mass, exp_mof_list = /
    import_experimental_data(exp_results_import, mof_list, mof_densities, gases)
sim_results_full = /
    import_simulated_data(sim_results_import, mof_list, mof_densities, gases)
element_pmf_results = /
    calculate_element_pmf(exp_results_full, sim_results_full, mof_list, stdev, mrange)
list_of_arrays, all_array_pmf_results = /
    calculate_all_arrays(mof_list, num_mofs, element_pmf_results, gases)
bins = /
    create_bins(gases, num_bins, mof_list, element_pmf_results)
binned_probabilities_sum, binned_probabilities_max = /
    bin_compositions(gases, bins, list_of_arrays, all_array_pmf_results)
array_kld_results = /
    calculate_kld(gases, list_of_arrays, bins, all_array_pmf_results, binned_probabilities_sum)
best_and_worst_arrays_by_absKLD, best_and_worst_arrays_by_jointKLD, best_and_worst_arrays_by_gasKLD = /
    choose_arrays(gases, num_mofs, array_kld_results, num_best_worst)

# --------------------------------------------------
# ----- Choose what to save ------------------------
# --------------------------------------------------
element_pmf_results_df = pd.DataFrame(data=element_pmf_results)
# save_element_pmf_data(element_pmf_results_df, stdev, mrange)
list_of_array_ids = assign_array_ids(list_of_arrays)
save_unbinned_array_pmf_data(gases, list_of_arrays, all_array_pmf_results)
save_binned_array_pmf_data(gases, list_of_arrays, list_of_array_ids, bins, binned_probabilities_sum)
plot_binned_array_pmf_data(gases, list_of_arrays, list_of_array_ids, bins, binned_probabilities_sum)
timestamp = (datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
write_data_as_tabcsv('saved_array_kld/best_and_worst_arrays_by_absKLD_%s.csv' % timestamp, best_and_worst_arrays_by_absKLD)
write_data_as_tabcsv('saved_array_kld/best_and_worst_arrays_by_jointKLD_%s.csv' % timestamp, best_and_worst_arrays_by_jointKLD)
write_data_as_tabcsv('saved_array_kld/best_and_worst_arrays_by_gasKLD_%s.csv' % timestamp, best_and_worst_arrays_by_gasKLD)
# write_data_as_tabcsv('saved_array_kld/all_ranked_by_jointKLD_%s.csv' % timestamp, all_ranked_by_jointKLD)
