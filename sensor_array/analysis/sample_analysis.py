# Code for caluclating the Henry's Constants for a variety of gas mixutres.

import copy
import csv
import itertools
import math
import numpy as np
import os
import pandas as pd
import random
import yaml

import tensorflow as tf
import tensorflow_probability as tfp


def yaml_loader(filepath):
    with open(filepath, 'r') as yaml_file:
        data = yaml.load(yaml_file)
    return data


def load_filter_unite_henrys_data(filepath, gases, min_comp=0.0):
    # ddf = dictionary of dataframes
    ddf = {gas: pd.read_csv("%s%s.csv" %(filepath, gas), sep='\t', engine='python') for gas in gases}
    ddf = {gas: ddf[gas][(np.isnan(ddf[gas].kh) == False) & (ddf[gas].max_comp >= 0.05)] for gas in gases}
    all_mofs = [set(ddf[gas]['MOF']) for gas in gases]
    common_mofs = all_mofs[0].intersection(*all_mofs[1::])
    ddf = {gas: ddf[gas][ddf[gas].MOF.isin(common_mofs)] for gas in gases}
    df_unified = pd.DataFrame(common_mofs, columns=['MOF'])

    for gas in gases:
        df_unified[gas] = list(ddf[gas]['gas_kh'])
        mass_constants = [list(ddf[gas]['pure_air_mass']) for gas in gases]
        avg_mass_constants = np.sum(mass_constants, axis=0)/np.shape(mass_constants)[0]
        df_unified['pure_air_mass'] = avg_mass_constants

    return df_unified, common_mofs


def create_uniform_comp_list(gases, gas_limits, spacing, filename=None, imply_final_gas_range=True, imply_final_gas_spacing=False, filter_for_1=True, round_at=None):
    """
    Function used to create a tab-delimited csv file of gas compositions for a set of gases with a
    range of compositions. The compositions of the final gas in the list is calculated so that the
    total mole fraction of the system is equal to 1 by default. This is true even if the composition
    is supplied, however this behavior can be turned off, in which case the list will contain only
    compositions in the given range which total to 1.
    """

    # Calculate the valid range of compositions for the final gas in the list.
    if len(gases) == len(gas_limits)+1:
        lower_limit_lastgas = 1-np.sum([limit[1] for limit in gas_limits])
        if lower_limit_lastgas < 0:
            lower_limit_lastgas = 0
        upper_limit_lastgas = 1-np.sum([limit[0] for limit in gas_limits])
        if upper_limit_lastgas > 1:
            upper_limit_lastgas = 1
        gas_limits_new = [limit for limit in gas_limits]
        gas_limits_new.append([lower_limit_lastgas, upper_limit_lastgas])
    elif len(gases) == len(gas_limits):
        if imply_final_gas_range == True:
            lower_limit_lastgas = 1-np.sum([limit[1] for limit in gas_limits[:-1]])
            if lower_limit_lastgas < 0:
                lower_limit_lastgas = 0
            upper_limit_lastgas = 1-np.sum([limit[0] for limit in gas_limits[:-1]])
            if upper_limit_lastgas > 1:
                upper_limit_lastgas = 1
            gas_limits_new = [limit for limit in gas_limits[:-1]]
            gas_limits_new.append([lower_limit_lastgas, upper_limit_lastgas])
        else:
            gas_limits_new = gas_limits

    # Determine the number of points for each gas for the given range and spacing.
    if len(spacing) == 1:
        number_of_values = [(limit[1]-limit[0])/spacing+1 for limit in gas_limits_new]
        number_of_values_as_int = [int(value) for value in number_of_values]
        if number_of_values != number_of_values_as_int:
            print('Bad combination of gas limits and spacing! Double check output file.')
        comps_by_gas = [np.linspace(gas_limits_new[i][0], gas_limits_new[i][1], number_of_values_as_int[i]) for i in range(len(gas_limits_new))]
        all_comps = list(itertools.product(*comps_by_gas))

    elif len(spacing) == len(gas_limits_new)-1:
        number_of_values = [(gas_limits_new[i][1]-gas_limits_new[i][0])/spacing[i]+1 for i in range(len(gas_limits_new)-1)]
        number_of_values_as_int = [int(value) for value in number_of_values]
        comps_by_gas = [np.linspace(gas_limits_new[i][0], gas_limits_new[i][1], number_of_values_as_int[i]) for i in range(len(gas_limits_new)-1)]
        all_comps_except_last = list(itertools.product(*comps_by_gas))
        all_comps = []
        for row in all_comps_except_last:
            total = np.sum(row)
            last_comp = 1 - total
            if last_comp >=0 and last_comp >= gas_limits_new[-1][0] and last_comp <= gas_limits_new[-1][1]:
                row += (last_comp,)
            all_comps.extend([row])

    elif len(spacing) == len(gas_limits_new):
        number_of_values = np.round([(gas_limits_new[i][1]-gas_limits_new[i][0])/spacing[i]+1 for i in range(len(gas_limits_new))], 5)
        number_of_values_as_int = [int(value) for value in number_of_values]
        if imply_final_gas_spacing == True:
            comps_by_gas = [np.linspace(gas_limits_new[i][0], gas_limits_new[i][1], number_of_values_as_int[i]) for i in range(len(gas_limits_new)-1)]
            all_comps_except_last = list(itertools.product(*comps_by_gas))
            all_comps = []
            for row in all_comps_except_last:
                total = np.sum(row)
                last_comp = 1 - total
                if last_comp >=0 and last_comp >= gas_limits_new[-1][0] and last_comp <= gas_limits_new[-1][1]:
                    row += (last_comp,)
                all_comps.extend([row])
        if imply_final_gas_spacing == False:
            if False in (number_of_values == number_of_values_as_int):
                print('Bad combination of gas limits and spacing! Double check output file.')
            comps_by_gas = [np.linspace(gas_limits_new[i][0], gas_limits_new[i][1], number_of_values_as_int[i]) for i in range(len(gas_limits_new))]
            all_comps = list(itertools.product(*comps_by_gas))

    # Filter out where total mole fractions != 1
    all_comps_final = []
    if filter_for_1 == True:
        for row in all_comps:
            if round_at != None:
                row = np.round(row, round_at)
            if np.sum(row) == 1:
                all_comps_final.extend([row])
    else:
        all_comps_final = all_comps

    # Write to file.
    if filename != None:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(gases)
            writer.writerows(all_comps_final)

    return comps_by_gas, all_comps_final


def create_pseudo_simulated_data_from_array(mof_list, comps_df, gases, kh_dataframe, append_to_df=False):
    pure_air_masses = [float(kh_dataframe.loc[kh_dataframe['MOF'] == mof]['pure_air_mass']) for mof in mof_list]
    khs = [[float(kh_dataframe.loc[kh_dataframe['MOF'] == mof][gas]) for gas in gases] for mof in mof_list]
    # rows_temp = [row for row in np.transpose([list(comps[gas]) for gas in gases])]
    rows_temp = comps_df.loc[:, gases].values.tolist()
    # rows_temp = [[comps.loc[i][gas] for gas in gases] for i in range(len(comps))]
    simulated_masses = [pure_air_masses + np.sum(np.multiply(khs, row), axis=1) for row in rows_temp]

    if append_to_df == True:
        for i in range(len(mof_list)):
            comps_df[mof_list[i]+'_mas'] = np.transpose(simulated_masses)[i]

    return simulated_masses, comps_df


def calculate_element_and_array_pmf_tf(simulated_masses, breath_sample_masses, comps, array, std_dev=0.01, append_to_df=False):

    distributions = tfp.distributions.TruncatedNormal(simulated_masses, std_dev, 0, np.inf)
    element_pmfs = distributions.prob(breath_sample_masses)
    element_norm_factors = 1/np.sum(element_pmfs, axis=0)
    element_pmfs_normalized = np.multiply(element_norm_factors, element_pmfs)

    # nnepmf = non-normalized element pmfs
    # nepmf = normalized element pmfs
    array_pmfs_nnepmf = np.prod(element_pmfs, axis=1)
    array_pmfs_nepmf = np.prod(element_pmfs_normalized, axis=1)
    array_norm_factor = 1/np.sum(array_pmfs_nepmf, axis=0)
    array_pmfs_normalized = np.multiply(array_norm_factor, array_pmfs_nepmf)

    if append_to_df == True:
        for i in range(len(array)):
            comps[array[i]+'_pmf'] = np.transpose(element_pmfs_normalized)[i]
        comps['array_pmf'] = array_pmfs_normalized

    # array_pmfs_nnepmf_sorted = sorted(array_pmfs_nnepmf, reverse=True)
    array_pmfs_normalized_sorted = sorted(array_pmfs_normalized, reverse=True)
    sorted_indicies = list(reversed(np.argsort(array_pmfs_normalized)))
    # sorted_indicies are NOT necessarily the same between nnempf_sorted and normalized_sorted since the elements are essentially given different "weights" by not normalizing their total probabilities.

    return comps, sorted_indicies


def subdivide_grid_from_array(comps, gases, spacing):

    new_grid_points = []
    for point in comps.values.tolist():
        new_points_by_component = []
        for i in range(len(point)):
            temp_set = []
            temp_set.extend([point[i]])
            if point[i]+spacing[i] <= 1:
                temp_set.extend([point[i]+spacing[i]])
            if point[i]-spacing[i] >= 0:
                temp_set.extend([point[i]-spacing[i]])
            new_points_by_component.extend([temp_set])
        new_grid_points_temp = list(itertools.product(*new_points_by_component))
        new_grid_points.extend(new_grid_points_temp)

    # Remove Duplicate Points
    new_grid_points = list(set(new_grid_points))
    new_grid_points = pd.DataFrame(new_grid_points, columns=gases)

    return new_grid_points


def check_prediciton_of_known_comp(henrys_data_no_air_effect, henrys_data_only_air_effect, henrys_data_combo, gases, mof_list, known_comp):

    # Create set of gases without Air
    gases_no_air = [gas for gas in gases if gas != 'Air']

    # Calculate the predicted mass (total and by component)
    predicted_mass = {}
    for mof in mof_list:
        temp_dict={}
        mass_temp = 0
        for gas in gases:
            # Get adsorbed mass of pure air mixture
            if gas == 'Air':
                mass_temp += henrys_data_combo[mof]['Pure Air Mass']
                temp_dict['Pure Air Mass'] = henrys_data_combo[mof]['Pure Air Mass']
                mass_temp_air_only = henrys_data_combo[mof]['Pure Air Mass']
                # Remove air displaced by other components
                for gas_2 in gases_no_air:
                    mass_temp_air_only += -henrys_data_only_air_effect[mof][gas_2+'_kH']*known_comp[gas_2+'_comp']
                temp_dict[gas+'_mass'] = mass_temp_air_only
            # Get adsorbed mass of Henry's Gases
            else:
                mass_temp += henrys_data_combo[mof][gas+'_kH']*known_comp[gas+'_comp']
                temp_dict[gas+'_mass'] = henrys_data_no_air_effect[mof][gas+'_kH']*known_comp[gas+'_comp']
            temp_dict['Total_mass'] = mass_temp

        predicted_mass[mof] = temp_dict

    return predicted_mass


def format_predicted_mass_as_breath_sample(predicted_mass, true_comp, run_id, random_error=False, random_seed=1):
    breath_sample = {}
    breath_sample['Run ID New'] = run_id
    np.random.seed(random_seed)
    for key in predicted_mass.keys():
        breath_sample[key] = predicted_mass[key]['Total_mass']
        if random_error != False:
            breath_sample[key+'_error'] = np.random.uniform(-1*random_error, random_error)
            breath_sample[key] += breath_sample[key+'_error']
        else:
            breath_sample[key+'_error'] = 0.0
    for key in true_comp.keys():
        breath_sample[key] = true_comp[key]

    return breath_sample


def comps_to_dataframe(comps, gases):
    df = pd.DataFrame(comps, columns=gases)

    return df


def load_breath_samples(filename):
    breath_samples = pd.read_csv(filename, sep='\t', engine='python')

    return breath_samples


def get_true_composoition(breath_sample, gases):
    # Isolate breath sample composition
    true_comp = {}
    for gas in gases:
        true_comp[gas+'_comp'] = float(breath_sample[gas+'_comp'])
    true_comp['Air_comp'] = true_comp['N2_comp']+true_comp['O2_comp']
    del true_comp['N2_comp']
    del true_comp['O2_comp']

    return true_comp


def calculate_all_arrays_list(mof_list, num_mofs):
    mof_array_list = []
    mof_array_list.extend(list(itertools.combinations(mof_list, num_mofs)))

    return mof_array_list


def composition_prediction_algorithm_new(array, henrys_data, gases, comps, spacing, convergence_limits, breath_sample_masses, num_cycles=10, fraction_to_keep=0.037, std_dev=0.10):

    # Initialize all values
    cycle_nums = [0]
    all_comp_sets = {gas:[] for gas in gases}
    final_comp_set = {gas:[] for gas in gases}
    convergence_status = {gas: False for gas in gases if gas !='Air'}

    # Record Initial Composition Range
    #Get min/max component mole frac
    all_molefrac = comps.values
    min_molefrac = all_molefrac.min(axis=0)
    max_molefrac = all_molefrac.max(axis=0)
    for g in range(len(gases)):
        gas = gases[g]
        all_comp_sets[gas].extend([[min_molefrac[g], max_molefrac[g]]])

    for i in range(num_cycles):
        # Keep track of cycles
        print('Cycle = ',i+1)
        print('Number of Comps. =', len(comps))
        cycle_nums.extend([i+1])

        # Convert from composition space to mass space to probability space
        print('\tCreate Pseudo-simulated Data...')
        # henrys_data = henrys_data_array
        simulated_masses, comps = create_pseudo_simulated_data_from_array(array, comps, gases, henrys_data, append_to_df=True)

        print('\tCalculating Element / Array Probability')
        comps, sorted_indicies = calculate_element_and_array_pmf_tf(simulated_masses, breath_sample_masses, comps, array, std_dev=std_dev, append_to_df=True)

        # Filter Out Low-Probability Compositions
        print('\tFiltering Low-probability Compositions.')
        filtered_indicies = sorted_indicies[0:int(np.ceil(fraction_to_keep*len(sorted_indicies)))]
        filtered_comps = comps.loc[filtered_indicies, gases]
        print('\tNumber of Comps. after filtering = ', len(filtered_comps))

        # Check / Update convergence status
        all_molefrac = filtered_comps.values
        min_molefrac = all_molefrac.min(axis=0)
        max_molefrac = all_molefrac.max(axis=0)
        molefrac_diff = max_molefrac-min_molefrac
        for g in range(len(gases)):
            gas = gases[g]
            all_comp_sets[gas].extend([[min_molefrac[g], max_molefrac[g]]])

            # Check Convergence
            final_comp_set[gas] = [min_molefrac[g], max_molefrac[g]]
            if gas != 'Air':
                if molefrac_diff[g] <= convergence_limits[gas]:
                    convergence_status[gas] = True
                else:
                    convergence_status[gas] = False

        # Check if exiting, Determine exit condition
        # Optipns are:
        #   (1) Max Number of Cycles Reached
        #   (2) All gases determined within desired range
        if False not in convergence_status.values() or i >= num_cycles-1:
            if False not in convergence_status.values() and i >= num_cycles-1:
                exit_condition = 'Compositions Converged & Maximum Number of Cycles Reached.'
            elif False not in convergence_status.values():
                exit_condition = 'Compositions Converged.'
            elif i >= num_cycles-1:
                exit_condition = 'Maximum Number of Cycles Reached.'

            print('Converged - Exiting!\n\n')

            return final_comp_set, exit_condition, cycle_nums, all_comp_sets

        else:
            print('\tSubdividing Grid...\n')
            spacing = [value*0.5 for value in spacing]
            comps = subdivide_grid_from_array(filtered_comps, gases, spacing)


def execute_sample_analysis(config_file):
    data = yaml_loader(config_file)

    materials_config_file = data['materials_config_filepath']
    materials_data = yaml_loader(materials_config_file)
    mof_list = materials_data['mof_list']

    gases_full = data['gases']
    gases = list(data['gases'].keys())
    henrys_data_filepath = data['henrys_data_filepath']
    breath_samples_filepath = data['breath_samples_filepath']
    convergence_limits = {gas: gases_full[gas]['convergence_limits'] for gas in gases}
    init_composition_limits = [gases_full[gas]['init_composition_limits'] for gas in gases]
    init_composition_spacing = [gases_full[gas]['init_composition_spacing'] for gas in gases]

    array = data['array']
    array_size = data['array_size']
    array_index = data['array_index']
    sample_types = data['sample_types']
    true_comp_at_start = data['true_comp_at_start']
    breath_samples_variation = data['breath_samples_variation']
    fraction_to_keep = data['fraction_to_keep']
    error_type_for_pmf = data['error_type_for_pmf']
    error_amount_for_pmf = data['error_amount_for_pmf']
    num_samples_to_test = data['num_samples_to_test']
    num_cycles = data['num_cycles']
    added_error_value = data['added_error_value']
    seed_value = data['seed_value']
    results_filepath = data['results_filepath']

    # ----- Create filepath if it does not exist -----
    if os.path.exists(results_filepath) != True:
        os.mkdir(results_filepath)


    # ----- Load Henry's Coefficient Data ----
    kh_dataframe, _ = load_filter_unite_henrys_data(henrys_data_filepath, gases, min_comp=0.0)
    henrys_data = kh_dataframe
    mof_list = common_mofs

    # ----- Create initial grid of points as a dictionary -----
    comps_by_component, comps_raw = create_uniform_comp_list(gases, init_composition_limits, init_composition_spacing, imply_final_gas_range=False, filter_for_1=False, round_at=None)
    comps = comps_to_dataframe(comps_raw, gases)

    # ----- Determine array if necessary -----
    if array == None:
        list_of_arrays = calculate_all_arrays_list(mof_list, array_size)
        # array = list_of_arrays[array_index]
        array = list_of_arrays[0]

    henrys_data_array = pd.DataFrame([kh_dataframe.loc[kh_dataframe['MOF'] == mof].values[0] for mof in array], columns=kh_dataframe.columns)


    # ----- Run Prediction Algorithm for Breath Samples! -----

    print('Beginning Analysis!')
    print('Convergence Limits = ', convergence_limits)

    # Clean up this loop
    for sample_type in sample_types:

        # ----- Load Breath Samples -----
        all_breath_samples = load_breath_samples(breath_samples_filepath)

        # ========== Limit Breath Sample Range for Testing ==========
        all_breath_samples = all_breath_samples.loc[0:num_samples_to_test-1]

        results_filename = 'breath_sample_prediciton_'+sample_type+'.csv'
        results_fullpath = results_filepath+results_filename
        sample_filename = 'breath_sample_'+sample_type+'.csv'
        sample_fullpath = results_filepath+sample_filename
        settings_filename = 'settings_'+sample_type+'.csv'
        settings_fullpath = results_filepath+settings_filename

        with open(settings_fullpath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Sample Types = ', sample_type])
            writer.writerow(['Sample Modifications = ', breath_samples_variation])
            writer.writerow(['Added Error = ', added_error_value, 'mg/g framework'])
            writer.writerow(['Random Seed for Error = ', seed_value])
            writer.writerow(['True Comp at Start = ', true_comp_at_start])
            writer.writerow(['Gas Limits = ', init_composition_limits])
            writer.writerow(['Gas Spacing = ', init_composition_spacing])
            writer.writerow(['Number of Initial Comps. = ', len(comps)])
            writer.writerow(['Fraction of Comps. Retained = ', fraction_to_keep])
            writer.writerow(['Error Type for Probability = ', error_type_for_pmf])
            writer.writerow(['Error Amount for Probability = ', error_amount_for_pmf])
            writer.writerow(['Convergence Limits = ', convergence_limits])

        # Make Folder to Add Plots to
        folder = sample_type+'/'
        os.mkdir(results_filepath+folder)

        # ----- Loop over all breath samples -----
        for i in range(len(all_breath_samples)):

            print('Breath Sample = ', i)

            # Load single breath sample
            breath_sample = all_breath_samples.loc[i:i]

            # Create copy of initial composition set, Add true comp explicitly if desired
            if true_comp_at_start == 'yes':
                comps = comps.append({gas: float(breath_sample[gas]) for gas in comps.keys()}, ignore_index=True)

            # Alter breath sample mass if desired
            breath_sample_masses, _ = create_pseudo_simulated_data_from_array(array, breath_sample, gases, kh_dataframe)
            if breath_samples_variation == 'almost perfect':
                breath_sample_masses +=  + np.random.normal(loc=0.0, scale=added_error, size=len(breath_sample_masses))

            final_comp_set, exit_condition, cycle_nums, all_comp_sets = composition_prediction_algorithm_new(array, henrys_data_array, gases, comps, init_composition_spacing, convergence_limits, breath_sample_masses, num_cycles=num_cycles, fraction_to_keep=fraction_to_keep, std_dev=error_amount_for_pmf)

            # Write Final Results to File
            if i == 0:
                with open(results_fullpath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(['Run ID', 'Exit Condition', 'Predicted Comp.', 'True Comp.'])
                    # writer.writerow([breath_sample['Run ID New']])
                    writer.writerow(['Exit Status = ', exit_condition])
                    writer.writerow([final_comp_set])
                    # writer.writerow([true_comp])
                with open(sample_fullpath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    # writer.writerow([breath_sample['Run ID New']])
                    writer.writerow([breath_sample])
                    writer.writerow([])
            else:
                with open(results_fullpath, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    # writer.writerow([breath_sample['Run ID New']])
                    writer.writerow(['Exit Status = ', exit_condition])
                    writer.writerow([final_comp_set])
                    # writer.writerow([true_comp])
                with open(sample_fullpath, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    # writer.writerow([breath_sample['Run ID New']])
                    writer.writerow([breath_sample])
                    writer.writerow([])

            # # Write and Plot Results for Single Breath Sample
            # full_sample_filepath = results_filepath+folder+'Sample_'+str(run_id)+'/'
            # os.mkdir(full_sample_filepath)
            #
            # # Write cycle results
            # cycle_results_filename = 'Sample'+str(run_id)+'.csv'
            # cycle_results_fullpath = full_sample_filepath+cycle_results_filename
            # with open(cycle_results_fullpath, 'w', newline='') as csvfile:
            #     writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            #     writer.writerow(['Cycle Nums.', *[gas for gas in gases]])
            #     for n in range(len(cycle_nums)):
            #       writer.writerow([cycle_nums[n], *[all_comp_sets[gas][n]for gas in gases]])
