# Code for caluclating the Henry's Constants for a variety of gas mixutres.

import ast
import copy
import csv
import glob
import itertools
import math
import numpy as np
import os
import random
import re
import scipy.stats as ss
import yaml

from collections import OrderedDict

import time
import tensorflow as tf
import tensorflow_probability as tfp


# ----- General Use -----
def yaml_loader(filepath):
    with open(filepath, 'r') as yaml_file:
        data = yaml.load(yaml_file)
    return data


def import_simulated_data(sim_results, sort_by_gas=False, gas_for_sorting=None):
    with open(sim_results) as file:
        reader = csv.DictReader(file, delimiter='\t')
        reader_list = list(reader)
        keys = reader.fieldnames

        for row in reader_list:
            # Isolate Mass Data since currently being assigned to single key
            mass_data_temp = [float(val) for val in row[keys[2]].split(' ')]
            num_gases = len(row)-len(mass_data_temp)-2
            # Reassign Compositions
            for i in range(num_gases):
                row[keys[-num_gases+i]] = row[keys[i+3]]
            # Reassign Masses
            for i in range(num_gases*2+2):
                row[keys[i+2]] = mass_data_temp[i]

        if sort_by_gas == True:
            reader_list = sorted(reader_list, key=lambda k: k[gas_for_sorting+'_comp'], reverse=False)

        return keys, reader_list


# ----- Prediciting Compositions -----
def load_henrys_data(figure_path, gases):
    data_hg_all = []
    data_air_all = []
    data_combo_all = []
    for gas in gases:
        filename_hg = '/Users/brian_day/Desktop/HC_Work/HenrysConstants_Analysis_Results/'+str(gas)+'_AllRatios/_henrys_coefficients_hg.csv'
        filename_air = '/Users/brian_day/Desktop/HC_Work/HenrysConstants_Analysis_Results/'+str(gas)+'_AllRatios/_henrys_coefficients_air.csv'
        data_hg = read_kH_results(filename_hg)
        data_air = read_kH_results(filename_air)
        data_hg_all.extend([data_hg])
        data_air_all.extend([data_air])

        data_combo_temp = []
        for row_hg in data_hg:
            row_combo_temp = {}
            for row_air in data_air:
                if row_hg['MOF'] == row_air['MOF']:
                    row_combo_temp['Gas'] = row_hg['Gas']
                    row_combo_temp['MOF'] = row_hg['MOF']
                    row_combo_temp['Maximum Composition'] = row_hg['Maximum Composition']
                    row_combo_temp['Pure Air Mass'] = row_air['Pure Air Mass']
                    if row_hg['k_H'] != None and row_air['k_H'] != None:
                        # Should be minus KH air since we fit it for increasing air, not increasing henry's gas (i.e. decreasing air).
                        row_combo_temp['k_H'] = row_hg['k_H']-row_air['k_H']
                    else:
                        row_combo_temp['k_H'] = None
            if row_combo_temp != {}:
                data_combo_temp.extend([row_combo_temp])
        data_combo_all.extend([data_combo_temp])

    return data_hg_all, data_air_all, data_combo_all


def filter_mof_list_and_average_pure_air_masses(mof_list, data_combo_all, min_allowed_comp=0.05):
    data_combo_avg = []
    for mof in mof_list:
        temp_collection = []
        for array in data_combo_all:
            for row in array:
                if row['MOF'].split('_')[0] == mof:
                    temp_collection.extend([row])

        max_comps = [row['Maximum Composition'] for row in temp_collection]

        if min(max_comps) < min_allowed_comp:
            continue

        pure_air_mass_all = []
        for row in temp_collection:
            if row['Pure Air Mass'] != None:
                pure_air_mass_all.extend([row['Pure Air Mass']])

        pure_air_mass_average = sum(pure_air_mass_all)/len(pure_air_mass_all)
        for row in temp_collection:
            if row['Pure Air Mass'] != None:
                row['Pure Air Mass'] = pure_air_mass_average

        data_combo_avg.extend([temp_collection])

    mof_list_filtered = [row[0]['MOF'].split('_')[0] for row in data_combo_avg]

    return mof_list_filtered, data_combo_avg


def reformat_henrys_data(mof_list_filtered, data_hg_all, data_air_all, data_combo_avg):
    data_hg_reformatted = {}
    for mof in mof_list_filtered:
        temp_dict_manual = {}
        for array in data_hg_all:
            for row in array:
                if row['MOF'].split('_')[0] == mof:
                    gas = row['Gas']
                    temp_dict_manual[gas+'_kH'] = row['k_H']
        data_hg_reformatted[mof] = temp_dict_manual

    data_air_reformatted = {}
    for mof in mof_list_filtered:
        temp_dict_manual = {}
        for array in data_air_all:
            for row in array:
                if row['MOF'].split('_')[0] == mof:
                    gas = row['Gas']
                    temp_dict_manual[gas+'_kH'] = row['k_H']
        data_air_reformatted[mof] = temp_dict_manual

    data_combo_reformatted = {}
    for mof in mof_list_filtered:
        temp_dict_manual = {}
        for array in data_combo_avg:
            for row in array:
                if row['MOF'].split('_')[0] == mof:
                    temp_dict_manual['Pure Air Mass'] = row['Pure Air Mass']
                    gas = row['Gas']
                    temp_dict_manual[gas+'_kH'] = row['k_H']
        data_combo_reformatted[mof] = temp_dict_manual

    return data_hg_reformatted, data_air_reformatted, data_combo_reformatted


def create_pseudo_simulated_data_from_array(mof_list, comps, gases, henrys_data):

    masses_pure_air = [henrys_data[mof]['Pure Air Mass'] for mof in mof_list]
    khs = [[henrys_data[mof][gas+'_kH'] for gas in gases] for mof in mof_list]
    simulated_masses = [masses_pure_air + np.sum(np.multiply(khs, row), axis=1) for row in comps]

    return simulated_masses


def calculate_element_and_array_pmf_tf(simulated_masses, breath_sample_masses, std_dev=0.01):

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

    array_pmfs_nnepmf_sorted = sorted(array_pmfs_nnepmf, reverse=True)
    array_pmfs_normalized_sorted = sorted(array_pmfs_normalized, reverse=True)
    sorted_indicies = list(reversed(np.argsort(array_pmfs_normalized)))
    # sorted_indicies are NOT necessarily the same between nnempf_sorted and normalized_sorted since the elements are essentially given different "weights" by not normalizing their total probabilities.

    return element_pmfs, element_pmfs_normalized, array_pmfs_nnepmf, array_pmfs_nnepmf_sorted, array_pmfs_normalized, array_pmfs_normalized_sorted, sorted_indicies


def subdivide_grid_from_array(comps, gases, spacing):

    new_grid_points = []
    for point in comps:
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

    return new_grid_points


def bin_compositions_by_convergence_status(comps, gases, convergence_status, array_pmf):
    # Determine which gases are used to create bins
    gases_to_bin_by = []
    for gas in gases:
        if gas != 'Air' and convergence_status[gas] == False:
            gases_to_bin_by.extend([gas])

    # Create bins and convert to a dictionary fromat for clarity
    all_comps_by_gas = {gas:[] for gas in gases_to_bin_by}
    for gas in gases_to_bin_by:
        all_comps_by_gas[gas] = list(set([comp[gas+'_comp'] for comp in comps]))
    bins = list(itertools.product(*[all_comps_by_gas[gas] for gas in gases_to_bin_by]))
    bins_as_dict = [{gases_to_bin_by[i]:bin[i] for i in range(len(gases_to_bin_by))} for bin in bins]

    # Convert bins to tuples, and create list for storing full composition set, and for element/array pmf of bins
    bins_as_keys = [tuple(bin.items()) for bin in bins_as_dict]
    comps_by_bins = {key:[] for key in bins_as_keys}
    pmf_by_bins = {key:[] for key in bins_as_keys}
    bin_pmf_by_bins = {key:{mof:0 for mof in array} for key in bins_as_keys}

    # Assign compositions to bins
    for row in comps:
        bin_key = tuple({gas:row[gas+'_comp'] for gas in gases_to_bin_by}.items())
        comps_by_bins[bin_key].extend([row])

    # Assign probability data to bins
    for row in array_pmf:
        bin_key = tuple({gas:row[gas+'_comp'] for gas in gases_to_bin_by}.items())
        pmf_by_bins[bin_key].extend([row])

    # Determine bin pmf
    for key in bins_as_keys:
        bin_pmf_by_bins[key]['Array_PMF'] = 0
        for row in pmf_by_bins[key]:
            for mof in array:
                bin_pmf_by_bins[key][mof] += row[mof+'_pmf']
            bin_pmf_by_bins[key]['Array_PMF'] += row['Array_pmf']

    return comps_by_bins, pmf_by_bins, bin_pmf_by_bins


def filter_binned_comps_by_probability(comps_by_bins, pmf_by_bins, bin_pmf_by_bins, gases, fraction_to_keep=0.037):

    # Sort bin pmfs, and filter
    bin_pmf_by_bins_sorted = sorted(bin_pmf_by_bins.items(), key=lambda item: item[1]['Array_PMF'], reverse=True)
    bins_to_keep = [item[0] for item in bin_pmf_by_bins_sorted[0:int(np.ceil(len(bin_pmf_by_bins_sorted)*fraction_to_keep**0.5))] ]

    # Unbin remaining compositions, and filter by comp. pmf
    array_pmf_after_bin_filtering = [comp for bin in bins_to_keep for comp in pmf_by_bins[bin]]
    array_pmf_after_bin_filtering_sorted = sorted(array_pmf_after_bin_filtering, key=lambda k: k['Array_pmf'], reverse = True)
    comps_to_keep = array_pmf_after_bin_filtering_sorted[0:int(np.ceil(len(array_pmf_after_bin_filtering_sorted)*fraction_to_keep**0.5))]
    comps_to_keep_clean = [ {gas+'_comp':row[gas+'_comp'] for gas in gases} for row in comps_to_keep ]

    return comps_to_keep_clean


def filter_unbinned_comps_by_probability(array_pmf_sorted, gases, fraction_to_keep=0.037):

    # Unbin remaining compositions, and filter by comp. pmf
    # array_pmf_sorted = sorted(array_pmf, key=lambda k: k['Array_pmf'], reverse = True)
    comps_to_keep = array_pmf_sorted[0:int(np.ceil(len(array_pmf_sorted)*fraction_to_keep))]
    comps_to_keep_clean = [ {gas+'_comp':row[gas+'_comp'] for gas in gases} for row in comps_to_keep ]

    return comps_to_keep_clean


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


def check_prediciton_of_known_comp_range(henrys_data_no_air_effect, henrys_data_only_air_effect, henrys_data_combo, gases, mof_list, known_comp):

    # Create set of gases without Air
    gases_no_air = [gas for gas in gases if gas != 'Air']

    # Calculate the predicted mass (total and by component)
    predicted_mass = {}
    for mof in mof_list:
        temp_dict={}
        mass_temp_lb = 0
        mass_temp_ub = 0
        for gas in gases:

            # Get adsorbed mass of pure air mixture
            if gas == 'Air':
                temp_dict['Pure Air Mass'] = henrys_data_combo[mof]['Pure Air Mass']
                mass_temp_lb += henrys_data_combo[mof]['Pure Air Mass']
                mass_temp_ub += henrys_data_combo[mof]['Pure Air Mass']
                mass_temp_air_only_lb = henrys_data_combo[mof]['Pure Air Mass']
                mass_temp_air_only_ub = henrys_data_combo[mof]['Pure Air Mass']

                # Remove air displaced by other components
                for gas_2 in gases_no_air:
                    mass_temp_air_only_lb += -henrys_data_only_air_effect[mof][gas_2+'_kH']*max(known_comp[gas_2+'_comp'])
                    mass_temp_air_only_ub += -henrys_data_only_air_effect[mof][gas_2+'_kH']*min(known_comp[gas_2+'_comp'])
                temp_dict[gas+'_mass'] = [mass_temp_air_only_lb, mass_temp_air_only_ub]

            # Get adsorbed mass of Henry's Gases
            else:
                mass_temp_lb += henrys_data_combo[mof][gas+'_kH']*min(known_comp[gas+'_comp'])
                mass_temp_ub += henrys_data_combo[mof][gas+'_kH']*max(known_comp[gas+'_comp'])

                temp_dict[gas+'_mass'] = list(henrys_data_no_air_effect[mof][gas+'_kH']*np.array(known_comp[gas+'_comp']))

            temp_dict['Total_mass'] = [mass_temp_lb, mass_temp_ub]

        predicted_mass[mof] = temp_dict

    return predicted_mass


def isolate_mass_by_component(predicted_mass, simulated_mass, gases, mof_list):
    # Initialize Dict
    adsorbed_masses = {gas:[] for gas in gases}
    adsorbed_masses_error = {gas:[] for gas in gases}
    predicted_masses = {gas:[] for gas in gases}

    # Get experimental masses and errors
    for mof in mof_list_filtered:
        for gas in gases:
            if gas == 'Air':
                adsorbed_masses[gas].extend([simulated_mass[mof]['N2_mass']+simulated_mass[mof]['O2_mass']])
                adsorbed_masses_error[gas].extend([simulated_mass[mof]['N2_error']+simulated_mass[mof]['O2_error']])
            else:
                adsorbed_masses[gas].extend([simulated_mass[mof][gas+'_mass']])
                adsorbed_masses_error[gas].extend([simulated_mass[mof][gas+'_error']])

    # Get predicted masses
    for mof in mof_list_filtered:
        for gas in gases:
            predicted_masses[gas].extend([predicted_mass[mof][gas+'_mass']])

    # Total mass for experimental data
    adsorbed_mass_total = np.zeros(len(adsorbed_masses[gases[0]]))
    adsorbed_mass_error_total = np.zeros(len(adsorbed_masses_error[gases[0]]))
    for gas in gases:
        adsorbed_mass_total = np.add(adsorbed_mass_total, adsorbed_masses[gas])
        adsorbed_mass_error_total = np.add(adsorbed_mass_error_total, adsorbed_masses_error[gas])
    adsorbed_masses['Total'] = list(adsorbed_mass_total)
    adsorbed_masses_error['Total'] = list(adsorbed_mass_error_total)

    # Total mass for predicted data
    if len(predicted_masses[gases[0]][0]) == 1:
        predicted_mass_total = np.zeros(len(predicted_masses[gases[0]]))
        for gas in gases:
            predicted_mass_total = np.add(predicted_mass_total, predicted_masses[gas])
        predicted_masses['Total'] = list(predicted_mass_total)

    else:
        predicted_mass_total = np.zeros([len(predicted_masses[gases[0]]),2])
        for gas in gases:
            predicted_mass_total = np.add(predicted_mass_total, predicted_masses[gas])
        predicted_masses['Total'] = list(predicted_mass_total)

    return adsorbed_masses, adsorbed_masses_error, predicted_masses


def comps_to_dict(comps, gases):
    comps_as_dict = []
    for row in comps:
        temp_dict = {}
        for i in range(len(gases)):
            temp_dict[gases[i]+'_comp'] = row[i]
        comps_as_dict.extend([temp_dict])

    return comps_as_dict


def load_breath_samples(breath_filepath, mof_list_filtered):
    # N.B. Somehow, only 48 diseased breath samples...
    # 2 Breath samples had negative concentrations, and thus never ran - Fix this in create comps
    files = list(glob.glob(breath_filepath+'*/*.csv'))

    # Create a set of all breath samples for all mofs (from filtered list)
    all_breath_samples = []
    for file in files:
        mof = re.split('/|_', file)[-2]
        if mof == 'v2':
            mof = mof = re.split('/|_', file)[-3]
        if mof in mof_list_filtered:
            _, breath_sample = import_simulated_data(file)
            all_breath_samples.extend(breath_sample)

    # Join results of the same breath sample
    num_samples = int(len(all_breath_samples)/len(mof_list_filtered))
    all_breath_samples_joined = []
    run_id_new = 0
    for row in all_breath_samples[0:num_samples]:

        # Create a temp_dict to add all mof data too
        temp_dict = {}
        run_id_new += 1
        row['Run ID New'] = run_id_new
        temp_dict['Run ID New'] = run_id_new
        gases = ['argon', 'ammonia', 'CO2', 'N2', 'O2']
        for gas in gases:
            temp_dict[gas+'_comp'] = row[gas+'_comp']

        # Create a comp_dict to see if same sample
        comp_dict = {}
        gases = ['argon', 'ammonia', 'CO2', 'N2', 'O2']
        for gas in gases:
            comp_dict[gas+'_comp'] = row[gas+'_comp']

        # Check all subsequent rows for match
        for row_2 in all_breath_samples:
            # row_count += 1
            comp_dict_2 = {}
            gases = ['argon', 'ammonia', 'CO2', 'N2', 'O2']
            for gas in gases:
                comp_dict_2[gas+'_comp'] = row_2[gas+'_comp']

            if comp_dict == comp_dict_2:
                row_2['Run ID New'] = run_id_new
                mof = row_2['MOF'].split('_')[0]
                temp_dict[mof] = row_2['total_mass']
                temp_dict[mof+'_error'] = row_2['total_mass_error']

        all_breath_samples_joined.extend([temp_dict])

    return all_breath_samples, all_breath_samples_joined


def load_breath_samples_alt(filename):
    gases = ['Argon', 'Ammonia', 'CO2', 'N2', 'O2']
    new_keys = {'Argon': 'argon_comp', 'Ammonia': 'ammonia_comp', 'CO2': 'CO2_comp', 'N2': 'N2_comp', 'O2': 'O2_comp'}

    with open(filename) as file:
        reader = csv.DictReader(file, delimiter='\t')
        reader_list = list(reader)
        for i in range(len(reader_list)):
            reader_list[i]['Run ID New'] = int(i+1)
            for gas in gases:
                reader_list[i][new_keys[gas]] = reader_list[i].pop(gas)
        keys = reader.fieldnames

    return keys, reader_list


def reload_full_breath_sample(breath_sample, all_breath_samples):
    breath_sample_full = []
    for row in all_breath_samples:
        if row['Run ID New'] == breath_sample['Run ID New']:
            breath_sample_full .extend([row])

    return breath_sample_full


def reformat_full_breath_sample(breath_sample_full):
    breath_sample_full_reformatted = {}
    for row in breath_sample_full:
        breath_sample_full_reformatted[row['MOF'].split('_')[0]] = row

    return breath_sample_full_reformatted


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


def import_prediction_data(prediction_results):
    with open(prediction_results) as file:
        reader = csv.reader(file, delimiter='\t')
        reader_list = list(reader)
        keys = reader_list[0]

        prediction_results = []
        for i in range(len(reader_list[1::])):
            if int(i+0) % 4 == 0:
                temp_dict = OrderedDict()
                temp_dict[keys[0]] = int(reader_list[i+1][0])
            elif int(i+3) % 4 == 0:
                temp_dict[keys[1]] = str(reader_list[i+1][0])
            elif int(i+2) % 4 == 0:
                temp_dict[keys[2]] = ast.literal_eval(reader_list[i+1][0])
            elif int(i+1) % 4 == 0:
                temp_dict[keys[3]] = ast.literal_eval(reader_list[i+1][0])
                prediction_results.extend([temp_dict])

        return keys, prediction_results


def composition_prediction_algorithm_new(array, henrys_data, gases, comps, spacing, convergence_limits, breath_sample_masses, num_cycles=10, fraction_to_keep=0.037, std_dev=0.10):

    # Initialize all values
    cycle_nums = [0]
    all_comp_sets = {gas:[] for gas in gases}
    final_comp_set = {gas:[] for gas in gases}
    convergence_status = {gas: False for gas in gases if gas !='Air'}

    all_array_pmfs_nnempf = []
    all_array_pmfs_normalized = []

    # Record Initial Composition Range
    for i in range(len(gases)):
        #Get min/max component mole frac
        gas = gases[i]
        all_molefrac = [comps[j][i] for j in range(len(comps))]
        min_molefrac = min(all_molefrac)
        max_molefrac = max(all_molefrac)
        all_comp_sets[gas].extend([[min_molefrac, max_molefrac]])

    for i in range(num_cycles):
        # Keep track of cycles
        print('Cycle = ',i+1)
        print('Number of Comps. =', len(comps))
        cycle_nums.extend([i+1])

        # Convert from composition space to mass space to probability space
        print('\tCreate Pseudo-simulated Data...')
        # start_time = time.time()
        simulated_masses = create_pseudo_simulated_data_from_array(array, comps, gases, henrys_data)
        # elapsed_time = time.time() - start_time
        # print('\t\tt =',elapsed_time,' s')

        print('\tCalculating Element / Array Probability')
        # start_time = time.time()
        _, element_pmfs_normalized, array_pmfs_nnepmf, array_pmfs_nnepmf_sorted, array_pmfs_normalized, array_pmfs_normalized_sorted, sorted_indicies = calculate_element_and_array_pmf_tf(simulated_masses, breath_sample_masses, std_dev=std_dev)
        # elapsed_time = time.time() - start_time
        # print('\t\tt =',elapsed_time,' s')

        # Log array_pmfs
        # Potentially a memory-hogging step which will need to use a temporary file to write to and read from.
        all_array_pmfs_nnempf.extend([array_pmfs_nnepmf_sorted])
        all_array_pmfs_normalized.extend([array_pmfs_normalized_sorted])

        # Filter Out Low-Probability Compositions
        print('\tFiltering Low-probability Compositions.')
        filtered_indicies = sorted_indicies[0:int(np.ceil(fraction_to_keep*len(sorted_indicies)))]
        filtered_comps = [comps[index] for index in filtered_indicies]
        print('\tNumber of Comps. after filtering = ', len(filtered_comps))

        # Check / Update convergence status
        for g in range(len(gases)):
            #Get min/max component mole frac
            gas = gases[g]
            all_molefrac = [filtered_comps[j][g] for j in range(len(filtered_comps))]
            min_molefrac = min(all_molefrac)
            max_molefrac = max(all_molefrac)
            molefrac_diff = max_molefrac-min_molefrac
            all_comp_sets[gas].extend([[min_molefrac, max_molefrac]])

            # Check Convergence
            final_comp_set[gas] = [min_molefrac, max_molefrac]
            if gas != 'Air':
                if molefrac_diff <= convergence_limits[gas]:
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

            return final_comp_set, exit_condition, cycle_nums, all_comp_sets, all_array_pmfs_nnempf, all_array_pmfs_normalized

        else:
            print('\tSubdividing Grid...\n')
            spacing = [value*0.5 for value in spacing]
            comps = subdivide_grid_from_array(filtered_comps, gases, spacing)


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


def read_kH_results(filename):
    with open(filename, newline='') as csvfile:
        output_data = csv.reader(csvfile, delimiter="\t")
        output_data = list(output_data)
        full_array = []
        for i in range(len(output_data)):
            row = output_data[i][0]
            row = row.replace('nan', '\'nan\'')
            row = row.replace('inf', '\'inf\'')
            row = row.replace('-\'inf\'', '\'-inf\'')
            temp_array = []
            temp_row = ast.literal_eval(row)
            # if type(temp_row['R^2']) == str or temp_row['R^2'] < 0:
            #     continue
            temp_array.append(temp_row)
            full_array.extend(temp_array)
        return full_array


def invert_matrix(array):
    if np.shape(array)[0] == np.shape(array)[1]:
        inv = np.linalg.inv(array)
    else:
        inv = np.linalg.pinv(array)

    return inv


def analytical_solution(array, gases, henrys_data_array, breath_sample_masses, added_error=None):
    pure_air_masses = [henrys_data_array[mof]['Pure Air Mass'] for mof in array]
    m_prime = [breath_sample_masses[i] - pure_air_masses[i] for i in range(len(breath_sample_masses))]
    henrys_matrix = [[henrys_data_array[mof][gas+'_kH'] for gas in gases] for mof in array]

    array_inv = invert_matrix(henrys_matrix)
    if added_error == None or added_error == 0:
        m_prime_new = m_prime
    else:
        m_prime_new = [value + random.uniform(-1,1)*1e-4 for value in m_prime]

    soln = np.matmul(array_inv, m_prime_w_error)
    soln_in_dict_format = {gases[i]+'_comp':soln[i] for i in range(len(gases))}

    return soln_in_dict_format


def calculate_KLD_for_cycle(array_pmfs):
    # This function requires NORMALIZED pmf values.

    num_points = len(array_pmfs)
    kld_max = math.log2(num_points)
    kld = sum( [float(pmf)*math.log2(float(pmf)*num_points) for pmf in array_pmfs if pmf != 0] )
    kld_norm = kld/kld_max

    return kld_norm


def calculate_p_max(num_elements, stddev):
    # This will be a (very close) approximations of p_max, since it is a truncated normal, and thus the value at the mean could change slightly subject to the contraint that the area under the curve, which goes to infinity, is exactly 1.
    distributions = tfp.distributions.TruncatedNormal(100, stddev, 0, np.inf)
    element_pmf_max = distributions.prob(100)
    array_pmf_max = element_pmf_max ** num_elements

    return array_pmf_max


def calculate_p_ratio(array_pmfs_sorted, p_max):
    """
    As long as each sensing element has a known associated std. dev. which is independent of composition, the maximum probability which could be assigned to a single point can be determined by mutliplyinf the individual max probabilities for each element. Thus, we can determine the ratio of the assigned probability of the maximum probability and use this as a metric to see how the predicition is improving.

    May need to adjust earlier function to report non-normalized element and/or array pmf.
    """
    all_p_ratios = [p_i/p_max for p_i in array_pmfs_sorted]

    return p_ratios


def execute_sample_analysis(config_file):
    config_file = 'config_files/sample_analysis_config_tests.yaml'
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
    data_hg_all, data_air_all, data_combo_all = load_henrys_data(henrys_data_filepath, gases)
    mof_list_filtered, data_combo_avg = filter_mof_list_and_average_pure_air_masses(mof_list, data_combo_all, min_allowed_comp=0.05)
    data_hg_reformatted, data_air_reformatted, data_combo_reformatted = reformat_henrys_data(mof_list_filtered, data_hg_all, data_air_all, data_combo_avg)
    henrys_data = data_combo_reformatted
    mof_list = mof_list_filtered

    # ----- Create initial grid of points as a dictionary -----
    comps_by_component, comps_raw = create_uniform_comp_list(gases, init_composition_limits, init_composition_spacing, imply_final_gas_range=False, filter_for_1=False, round_at=None)
    comps_as_dict = comps_to_dict(comps_raw, gases)

    # ----- Determine array if necessary -----
    if array == None:
        list_of_arrays = calculate_all_arrays_list(mof_list_filtered, array_size)
        array = list_of_arrays[array_index]

    henrys_data_array = {key: henrys_data[key] for key in array}


    # ----- Run Prediction Algorithm for Breath Samples! -----

    print('Beginning Analysis!')
    print('Convergence Limits = ', convergence_limits)

    # Clean up this loop
    for sample_type in sample_types:

        # ----- Load Breath Samples -----
        _, all_breath_samples_joined = load_breath_samples_alt(breath_samples_filepath)

        # ========== Limit Breath Sample Range for Testing ==========
        all_breath_samples_joined = all_breath_samples_joined[0:num_samples_to_test]

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
            writer.writerow(['Number of Initial Comps. = ', len(comps_as_dict)])
            writer.writerow(['Fraction of Comps. Retained = ', fraction_to_keep])
            writer.writerow(['Error Type for Probability = ', error_type_for_pmf])
            writer.writerow(['Error Amount for Probability = ', error_amount_for_pmf])
            writer.writerow(['Convergence Limits = ', convergence_limits])

        # Make Folder to Add Plots to
        folder = sample_type+'/'
        os.mkdir(results_filepath+folder)

        # ----- Loop over all breath samples -----
        for i in range(len(all_breath_samples_joined)):

            print('Breath Sample = ', i)

            # Load single breath sample
            breath_sample = all_breath_samples_joined[i]
            run_id = breath_sample['Run ID New']

            # Get true comp and predicted mass
            gases_for_true_comp = gases+['N2', 'O2']
            true_comp = get_true_composoition(breath_sample, gases_for_true_comp)
            true_comp_values = [true_comp[gas+'_comp'] for gas in gases]

            # Create copy of initial composition set, Add true comp explicitly if desired
            comps = copy.deepcopy(comps_raw)
            if true_comp_at_start == 'yes':
                comps.extend([true_comp_values])

            # Alter breath sample if desired
            gases_temp = gases+['Air']
            predicted_mass = check_prediciton_of_known_comp(data_hg_reformatted, data_air_reformatted, data_combo_reformatted, gases_temp, mof_list_filtered, true_comp)
            if breath_samples_variation == 'perfect':
                perfect_breath_sample = format_predicted_mass_as_breath_sample(predicted_mass, true_comp, run_id, random_error=False)
                breath_sample = perfect_breath_sample
            elif breath_samples_variation == 'almost perfect':
                almost_perfect_breath_sample = format_predicted_mass_as_breath_sample(predicted_mass, true_comp, run_id, random_error=added_error_value, random_seed=seed_value)
                breath_sample = almost_perfect_breath_sample
            breath_sample_masses = [breath_sample[mof] for mof in array]

            final_comp_set, exit_condition, cycle_nums, all_comp_sets, all_array_pmfs_nnempf, all_array_pmfs_normalized  = composition_prediction_algorithm_new(array, henrys_data_array, gases, comps, init_composition_spacing, convergence_limits, breath_sample_masses, num_cycles=num_cycles, fraction_to_keep=fraction_to_keep, std_dev=error_amount_for_pmf)

            # Write Final Results to File
            if i == 0:
                with open(results_fullpath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(['Run ID', 'Exit Condition', 'Predicted Comp.', 'True Comp.'])
                    writer.writerow([breath_sample['Run ID New']])
                    writer.writerow(['Exit Status = ', exit_condition])
                    writer.writerow([final_comp_set])
                    writer.writerow([true_comp])
                with open(sample_fullpath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([breath_sample['Run ID New']])
                    writer.writerow([breath_sample])
                    writer.writerow([])
            else:
                with open(results_fullpath, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([breath_sample['Run ID New']])
                    writer.writerow(['Exit Status = ', exit_condition])
                    writer.writerow([final_comp_set])
                    writer.writerow([true_comp])
                with open(sample_fullpath, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([breath_sample['Run ID New']])
                    writer.writerow([breath_sample])
                    writer.writerow([])

            # Write and Plot Results for Single Breath Sample
            full_sample_filepath = results_filepath+folder+'Sample_'+str(run_id)+'/'
            os.mkdir(full_sample_filepath)

            # Write cycle results
            cycle_results_filename = 'Sample'+str(run_id)+'.csv'
            cycle_results_fullpath = full_sample_filepath+cycle_results_filename
            with open(cycle_results_fullpath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Cycle Nums.', *[gas for gas in gases]])
                for n in range(len(cycle_nums)):
                    writer.writerow([cycle_nums[n], *[all_comp_sets[gas][n]for gas in gases]])
