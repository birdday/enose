# --------------------------------------------------
# ----- Import Python Packages ---------------------
# --------------------------------------------------
import sys
import pandas as pd
from datetime import datetime
from GeneticAlgorithm_BrianEdits import *

# --------------------------------------------------
# ----- Import RASPA Data and yaml File ------------
# --------------------------------------------------
# Redefine system arguments
# sim_data = sys.argv[1]
# exp_data = sys.argv[2]

# Data Paths
# 1 bar
# sim_data = '/Users/brian_day/Github-Repos/Sensor_Array/ALL_MOFS.csv'
# exp_data = '/Users/brian_day/Github-Repos/Sensor_Array/exp_data_175.csv'

# 5 and 10 bar
sim_data = '/Users/brian_day/Desktop/WilmerLab_Research/Research Projects/CO2 Sensing/CO2_RASPA_Results/CO2_5bar/CO2_Sensing_Results_New/ALL_MOFS.csv'
exp_data = '/Users/brian_day/Desktop/WilmerLab_Research/Research Projects/CO2 Sensing/CO2_RASPA_Results/CO2_5bar/CO2_Sensing_Results_New/exp_data_175.csv'

# Import results as dictionary
sim_results_import = read_data_as_dict(sim_data)
exp_results_import = read_data_as_dict(exp_data)

# Import yaml file as dictoncary
filepath = '/Users/brian_day/Github-Repos/Sensor_Array/settings/process_config.yaml'
data = yaml_loader(filepath)

# Redefine key varaibles in yaml file
num_mofs = data['number_mofs']
num_mixtures = data['num_mixtures']
num_bins = data['number_bins']
# num_best_worst = data['num_best_worst']
stdev = data['stdev']
mrange = data['mrange']
gases = data['gases']
mof_list = data['mof_array']
mof_densities = {}
for mof in mof_list:
    mof_densities.copy()
    mof_densities.update({ mof : data['mofs'][mof]['density']})

# --------------------------------------------------
# ----- Calculate all single MOF PMFs --------------
# --------------------------------------------------
(exp_results_full, exp_results_mass, exp_mof_list) = (
    import_experimental_data(exp_results_import, mof_list, mof_densities, gases) )
(sim_results_full) = (
    import_simulated_data(sim_results_import, mof_list, mof_densities, gases) )
(element_pmf_results) = (
    calculate_element_pmf(exp_results_full, sim_results_full, mof_list, stdev, mrange) )

# --------------------------------------------------
# ----- Run the Genetic Algorithm ------------------
# --------------------------------------------------

array_size_vect = [10, 15, 20, 25, 30, 35, 40, 45]
# array_size_vect = [1,2,3,4,5]
num_runs = 3
population_size = 20
num_generations = [25, 25, 50, 50, 50]
mutation_rates = [0.50, 0.20, 0.10, 0.05, 0.02]
seek = 'best'
seek_by = 'Absolute_KLD'
num_best = 2
num_lucky = 2
results_filepath = '/Users/brian_day/Desktop/GeneticAlgorithm_Results/elitist_GA/'

for array_size in array_size_vect:
    for i in range(num_runs):
        print('Beginning run #', i+1)
        for j in range(len(num_generations)):
            print('.')
            if j == 0:
                first_gen = create_first_generation(population_size, array_size, mof_list)
                gen_start_num = 1
                GA_array_list, GA_results = run_genetic_algorithm(first_gen, gen_start_num, \
                    array_size, mof_list, num_best, num_lucky, population_size, num_generations[0], mutation_rates[0], \
                    element_pmf_results, gases, num_bins, seek=seek, seek_by=seek_by)
            else:
                GA_array_list_temp = []
                GA_results_temp = []
                first_gen = GA_array_list[-1]
                gen_start_num = np.sum(num_generations[0:j])+1
                GA_array_list_temp, GA_results_temp = run_genetic_algorithm(first_gen, gen_start_num, \
                    array_size, mof_list, num_best, num_lucky, population_size, num_generations[j], mutation_rates[j], \
                    element_pmf_results, gases, num_bins, seek=seek, seek_by=seek_by)
                GA_array_list.extend(GA_array_list_temp)
                GA_results.extend(GA_results_temp)

        csv_filename = 'arraysize_%s_testnum_%s_%s_by_%s.csv' % (array_size, i+1, seek, seek_by)
        csv_fullname = results_filepath + csv_filename
        write_GA_results_clean(csv_fullname, GA_results)

        fig_filename = 'arraysize_%s_testnum_%s_%s_by_%s.png' % (array_size, i+1, seek, seek_by)
        fig_fullname = results_filepath + fig_filename
        plot_GA_results(fig_fullname, GA_results, num_generations, array_size, seek=seek, seek_by=seek_by)
