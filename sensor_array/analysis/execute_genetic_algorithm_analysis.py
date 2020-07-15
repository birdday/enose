# --------------------------------------------------
# ----- Import Python Packages ---------------------
# --------------------------------------------------
import sys
import pandas as pd
from datetime import datetime
from genetic_algorithm_analysis import *

# --------------------------------------------------
# ----- Import RASPA Data and yaml File ------------
# --------------------------------------------------
# Import yaml file as dictoncary
filepath = 'config_files/process_config.sample.yaml'
data = yaml_loader(filepath)

# Redefine key varaibles in yaml file
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
# ----- Calculate all single MOF PMFs --------------
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

# --------------------------------------------------
# ----- Run the Genetic Algorithm ------------------
# --------------------------------------------------

array_size_vect = [5, 10, 15, 20, 25, 30, 35, 40, 45]
num_runs = 1
population_size = 20
num_generations = [25, 25, 50, 50, 50]
mutation_rates = [0.50, 0.20, 0.10, 0.05, 0.02]
seek = 'best'
seek_by = 'Absolute_KLD'
num_best = 2
num_lucky = 2
results_filepath = '/Users/brian_day/Desktop/GeneticAlgorithm_Results/'

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
