"""
Original Code by: Jenna Gustafson
Modifications by: Brian Day

Jenna:
﻿This code imports mass adsorption data and 'experimental' adsorption data
from simulated output, for multiple MOFs and gas mixtures, and calculates the
probability that specific gases are present in specific mole fractions (in
each experimental case) based on the total mass adsorbed for each MOF.
Additionally, the best MOF arrays for detecting each gas are reported,
according to the highest information gain.

Brian:
Note on list comprehension: List comprehension will catch all instances, where
as simple for loops will only catch the last instance, unless all variables
are pre-allocated and appended to. Use list comprehension whenever possible.
In some instances of the code, using a mof twice would not be captured. Try to
fix this going forward.

--------------------------------------------------
----- Full List of Keywords used in functions ----
--------------------------------------------------
N.B. Keywords are also defined a second time with each function. Tried to use
the same keyword whenever the argument was the same for clarity.
  Simulation results:
    exp_results_import -- dictionary formatted experimental results
    sim_results_import -- dictionary formatted simulated results
  From yaml file:
    stdev -- standard deviation for the normal distribution
    mrange -- range for which the difference between CDFs is calculated
    gases -- list of gases in simulated mixtures
    num_bins -- number of bins specified by user in configuration file
    num_mixtures -- specify integer number of mixtures to add (for interpolation)
    num_mofs -- lower and upper limit of desired number of MOFs in array
    mof_list -- list of MOF structures simulated
    mof_densities -- dictionary of densities
  Calculated by functions / Used internally:
    mof_array -- list of MOFs in a single array
    list_of_arrays -- list of all arrays, and the MOFs in eaach array
    bins -- dictionary containing bin points for each gas
    binned_probabilities -- list of dictionaries, MOF array, gas, PMFs
    element_pmf_results -- list of dictionaries including MOF, mixture, probability
    array_pmf_results -- list of dictionaries, arrays, joint PMFs
    array_kld_results -- list of dictionaries including, MOF array, gas, and corresponding KLD
  Miscellaneous:
    comps -- list of all simulated gas compositions
"""

# --------------------------------------------------
# ----- Import Python Packages ---------------------
# --------------------------------------------------
import copy
import csv
import numpy as np
import operator
import os
import pandas as pd
import scipy.interpolate as si
import scipy.stats as ss
import sys
import yaml

from datetime import datetime
from functools import reduce
from itertools import combinations
from math import isnan, log
from matplotlib import pyplot as plt
from random import random
from scipy.spatial import Delaunay
from scipy.interpolate import spline

# --------------------------------------------------
# ----- User-defined Python Functions --------------
# --------------------------------------------------

# ----- Read and Write Data Files -----
def read_data_as_dict(filename):
    with open(filename,newline='') as csvfile:
        output_data = csv.DictReader(csvfile, delimiter="\t")
        return list(output_data)


def write_data_as_tabcsv(filename, data):
    with open(filename,'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        for line in data:
            writer.writerow([line])
    return writer


def yaml_loader(filepath):
    with open(filepath, 'r') as yaml_file:
        data = yaml.load(yaml_file)
    return data


def import_experimental_data(exp_results_import, mof_list, mof_densities, gases):
    """
    ----- Convert simulated/experimental data into dictionary format -----
    Also calculates masses in terms of mg/(cm^3 of framework)
    Keyword arguments:
        exp_results_import -- dictionary formatted experimental results
        sim_results_import -- dictionary formatted simulated results
        mof_list -- list of MOF structures simulated
        mof_densities -- dictionary of densities
        gases -- list of gases in simulated mixtures
    Can probibily omit exp_results_mass in the future and just use the full
    dictionary for further analysis.
    """
    exp_results_full = []
    exp_results_mass = []
    exp_mof_list = []
    for mof in mof_list:
        for row in exp_results_import:
            if row['MOF'] == mof:
                mass = float(mof_densities[mof]) * float(row['Mass'])
                exp_results_temp = row.copy()
                exp_results_temp.update({'Mass_mg/cm3' : mass})
                exp_results_full.extend([exp_results_temp])
                exp_results_mass.append({'MOF' : mof, 'Mass' : mass})
                exp_mof_list.append(str(mof))
            else:
                None
    return exp_results_full, exp_results_mass, exp_mof_list


def import_simulated_data(sim_results_import, mof_list, mof_densities, gases):
    sim_results_full = []
    for mof in mof_list:
        for row in sim_results_import:
            if row['MOF'] == mof:
                mass = float(mof_densities[mof]) * float(row['Mass'])
                sim_results_temp = row.copy()
                sim_results_temp.update({'Mass_mg/cm3' : mass})
                sim_results_full.extend([sim_results_temp])
    return sim_results_full


def add_random_gas(gases, comps, num_mixtures):
    """
    ----- Interpolate between simulated data points -----
    ===== NOT FUNCTIONAL AT THE MOMENT =====
    Adds gas mixtures to the original data, between min and max of original mole fractions, as the
    code only predicts simulated mixtues. Interpolation can improve accuracy of prediciton.
    Currently designed for ternary mixtures. Change later if needed.
    Keyword arguments:
        comps -- all simulated gas compositions
        num_mixtures -- specify integer number of mixtures to add
    """
    d0_range = [min(comps[:,0]), max(comps[:,0])]
    d1_range = [min(comps[:,1]), max(comps[:,1])]
    d2_range = [min(comps[:,2]), max(comps[:,2])]
    while (len(comps) < 78 + num_mixtures):
        predicted_mass = interp_dat(random_gas)
        if sum(random_gas) <= 1 and not isnan(predicted_mass):
            comps.append(random_gas)
            masses.extend(predicted_mass)


def calculate_element_pmf(exp_results_full, sim_results_full, mof_list, stdev, mrange):
    """
    ----- Calculates probability mass function (PMF) of each data point -----
    Keyword arguments:
        exp_results_full -- dictionary formatted experimental results (with new mass units)
        sim_results_full -- dictionary formatted experimental results (with new mass units)
        mof_list -- names of all MOFs
        stdev -- standard deviation for the normal distribution
        mrange -- range for which the difference between cdfs is calculated
    ----------
    Formula for the probability of x in the trunacted PDF over range [a,b] is:
                              Norm_PDF(mean,var,x)
    Trunc_PDF(x) = -------------------------------------------
                   Norm_CDF(mean,var,b) - Norm_CDF(mean,var,a)
    Can be calculated directly in python with:
      Trunc_PDF(x) = ss.truncnorm.pdf(x, alpha, beta, loc=mu, scale=sigma)
    Jenna's approach is probably better. She calculates the probaility that that
    the real mass falls in a small range around the measured mass. Examine the
    difference between these two approaches in more detail, along with the effects
    of the employed error function parameters.
    """
    element_pmf_results = []
    for mof in mof_list:
        # Isolate the simulated results for the mof
        all_results_sim = [row for row in sim_results_full if row['MOF'] == mof]
        all_masses_sim = [row['Mass_mg/cm3'] for row in all_results_sim]

        # Isolate the experimental result(s) for the mof
        all_results_exp = [row for row in exp_results_full if row['MOF'] == mof]
        all_masses_exp = [row['Mass_mg/cm3'] for row in all_results_exp]

        # Calculate all pmfs based on the experimental mass and truncated normal
        # probability distribution.
        mof_temp_dict = []
        for mass_exp in all_masses_exp:
            probs_range = []
            # probs_exact = []
            a, b = 0, float(max(all_masses_sim)) * (1 + mrange)
            mu, sigma = float(mass_exp), float(stdev)*float(mass_exp)
            alpha, beta = ((a-mu)/sigma), ((b-mu)/sigma)
            for mass_sim in all_masses_sim:
                upper_prob = ss.truncnorm.cdf(float(mass_sim) * (1 + mrange), alpha, beta, loc = mu, scale = sigma)
                lower_prob = ss.truncnorm.cdf(float(mass_sim) * (1 - mrange), alpha, beta, loc = mu, scale = sigma)
                probs_range.append(upper_prob - lower_prob)
                # prob_singlepoint = ss.truncnorm.pdf(float(mass_sim), alpha, beta, loc=mu, scale=sigma)
                # probs_exact.append(prob_singlepoint)
            sum_probs_range = sum(probs_range)
            norm_probs_range = [(i/sum_probs_range) for i in probs_range]
            # sum_probs_exact = sum(probs_exact)
            # norm_probs_exact = [(i/sum_probs_exact) for i in probs_exact]

            # Update dictionary with pmf for each MOF
            new_temp_dict = []
            # Initialize the dictioary
            if mof_temp_dict == []:
                for index in range(len(norm_probs_range)):
                    mof_temp_dict = all_results_sim[index].copy()
                    mof_temp_dict.update({ 'PMF_Range' : norm_probs_range[index] })
                    # mof_temp_dict.update({ 'PMF_Exact' : norm_probs_exact[index] })
                    new_temp_dict.extend([mof_temp_dict])
                new_temp_dict_2 = new_temp_dict
            # Add to the exisitng dictionary
            else:
                for index in range(len(norm_probs_range)):
                    mof_temp_dict = new_temp_dict_2[index].copy()
                    mof_temp_dict.update({ 'PMF_Range' : norm_probs_range[index] })
                    # mof_temp_dict.update({ 'PMF_Exact' : norm_probs_exact[index] })
                    new_temp_dict.extend([mof_temp_dict])
                new_temp_dict_2 = new_temp_dict

        element_pmf_results.extend(new_temp_dict_2)

    return element_pmf_results


def calculate_array_pmf(mof_array, element_pmf_results):
    """
    ----- Combines and normalizes mof pmfs for a single array -----
    Function is used in function 'calculate_all_arrays' below
    Keyword arguments:
        mof_array -- list of mofs in a single array
        element_pmf_results -- list of dictionaries including mof, mixture, probability
    """
    compound_pmfs = None
    for mof in mof_array:
        mof_pmf = [ row['PMF_Range'] for row in element_pmf_results if row['MOF'] == mof ]
        if compound_pmfs is not None:
            compound_pmfs = [x*y for x,y in zip(compound_pmfs, mof_pmf)]
        else:
            compound_pmfs = mof_pmf
    norm_factor = sum(compound_pmfs)
    single_array_pmf_results = [ i / norm_factor for i in compound_pmfs ]
    return single_array_pmf_results


def calculate_all_arrays(mof_list, num_mofs, element_pmf_results, gases):
    """
    ----- Calculates all possible arrays and corresponding pmfs ----
    Sets up all combinations of MOF arrays, uses function 'calculate_array_pmf'
    to get pmf values for every array/gas/experiment combination
    Keyword arguments:
        mof_list -- list of all mofs
        num_mofs -- lower and upper limit of desired number of mofs in array
        element_pmf_results -- list of dictionaries including mof, mixture, probability
        gases -- list of gases
    """

    # Creates list of MOF arrays, all combinations from min to max number of MOFs
    mof_array_list = []
    array_size = min(num_mofs)
    while array_size <= max(num_mofs):
        mof_array_list.extend(list(combinations(mof_list, array_size)))
        array_size += 1

    # Save list of dictionaries for first MOF for use as a list of all gas mole fractions
    comp_set_dict = [row for row in element_pmf_results if row['MOF'] == mof_list[0]]

    # Calculate and save the pmfs for each of the generated arrys
    all_array_pmf_results = []
    for mof_array in mof_array_list:
        single_array_pmf_results = calculate_array_pmf(mof_array, element_pmf_results)
        if mof_array == mof_array_list[0]:
            # First Array: Set up dictionary with keys
            for index in range(len(comp_set_dict)):
                array_dict = {'%s' % ' '.join(mof_array) : single_array_pmf_results[index]}
                for gas in gases:
                    array_dict.update({ '%s' % gas : float(comp_set_dict[index][gas])})
                all_array_pmf_results.extend([array_dict])
        else:
            # Not First Array: Update Dictionary
            for index in range(len(comp_set_dict)):
                all_array_pmf_results[index]['%s' % ' '.join(mof_array)] = single_array_pmf_results[index]

    return mof_array_list, all_array_pmf_results


def create_bins(gases, num_bins, mof_list, element_pmf_results):
    """
    ----- Creates bins for all gases -----
    Keyword arguments:
        gases -- list of present gases
        num_bins -- number of bins specified by user in config file
        mof_list -- list of mofs used in analysis
        element_pmf_results -- list of dictionaries including mof, mixture, probability
    """

    # Save list of dictionaries for first MOF to use as a list of all gas mole fractions
    # Might be good to make this it's own function early on for easier access to list of comps
    comp_set_dict = [row for row in element_pmf_results if row['MOF'] == mof_list[0]]
    # comps_array = []
    # for row in comp_set_dict:
    #     comps_array.append([float(row[gas]) for gas in gases])
    comps_array = np.array([[float(row[gas]) for gas in gases] for row in comp_set_dict])
    # Figure out what is different between commented approach and current one!

    # Determine the set of points used to create bins
    bin_points = []
    for i in range(len(gases)):
        lower_comp = min(comps_array[:,i])
        upper_comp = max(comps_array[:,i])
        # Brian Bins
        lower_lim = lower_comp - 0.5*(upper_comp-lower_comp)/(num_bins-1)
        upper_lim = upper_comp + 0.5*(upper_comp-lower_comp)/(num_bins-1)
        # Jenna Bins
        # lower_lim = lower_comp
        # upper_lim = upper_comp + (upper_comp-lower_comp)/(num_bins)
        bin_points_temp = np.linspace(lower_lim, upper_lim, num=num_bins+1, endpoint=True)
        # Extra steps to deal with impercision of values generated by np.linspace
        bin_points_temp = np.round(bin_points_temp,3)-(1e-4)
        bin_points_temp = np.round(bin_points_temp,4)
        bin_points.append([bin_points_temp])
    bin_points = np.transpose(np.vstack(bin_points))

    # Reformat bin_points
    bins = []
    for row in bin_points:
        bins.append({gases[i] : row[i] for i in range(len(gases))})

    return bins

def bin_compositions(gases, bins, list_of_arrays, all_array_pmf_results):
    """
    ----- Sorts pmfs into bins created by create_bins function -----
    The goal of this fucntion is to take the pmfs over the whole composition space
    and determine the probability vs. composition for one single gas, independent
    of the other componentns and their compositions.
    The current approach might be flawed in that it can skew the net probability
    based on the number of points in a bin. Revisit this in the near future.
    Keyword arguments:
        gases -- list of gases specified as user input
        list_of_arrays -- list of all array combinations
        bins -- dictionary containing bins for each gas
        all_array_pmf_results -- list of dictionaries, arrays, joint pmfs
    """

    binned_probabilities_sum = []
    binned_probabilities_max = []
    for gas in gases:
        # Assigns pmf to bin value (dictionary) by checking whether mole frac is
        # between the current and next bin value.
        # Current method likely inefficient, but it works. Can revisit later.
        for row in all_array_pmf_results:
             for i in range(1, len(bins)):
                 lower_bin = bins[i-1][gas]
                 upper_bin = bins[i][gas]
                 gas_comp = float(row[gas])
                 if gas_comp >= lower_bin and gas_comp < upper_bin:
                     row.update({'%s bin' % gas: lower_bin})

        # Loops through all of the bins and takes sum over all pmfs in that bin.
        all_bins_temp_sum = []
        all_bins_temp_max = []
        array_names = ['%s' % ' '.join(array) for array in list_of_arrays]
        for bin in bins[0:len(bins)-1]:
            array_pmfs_temp = {array: [] for array in array_names}
            for row in all_array_pmf_results:
                if bin[gas] == row['%s bin' % gas]:
                    for array in array_names:
                        array_pmfs_temp[array].append(row[array])

            # Updates pmfs for each array for current bin
            # Many methods here of handling multiple data points.
            # Currently, can sum all pmfs, or take the max value.
            single_bin_temp = {'%s bin' % gas : bin[gas]}
            with_sum = copy.deepcopy(single_bin_temp)
            with_max = copy.deepcopy(single_bin_temp)
            for array in array_names:
                if array_pmfs_temp[array] == []:
                    with_sum.update({array : 0})
                    with_max.update({array : 0})
                else:
                    with_sum.update({array : sum(array_pmfs_temp[array])})
                    with_max.update({array : max(array_pmfs_temp[array])})
            all_bins_temp_sum.append(with_sum)
            all_bins_temp_max.append(with_max)

        # Creates list of binned probabilities, already normalized
        binned_probabilities_sum.extend(all_bins_temp_sum)
        binned_probabilities_max.extend(all_bins_temp_max)

    return binned_probabilities_sum, binned_probabilities_max


def calculate_kld(gases, list_of_arrays, bins, all_array_pmf_results, binned_probabilities):
    """
    ----- Calculates the Kullback-Liebler Divergence of a MOF array with each gas component -----
    Keyword arguments:
        gases -- list of gases specified by user
        list_of_arrays -- list of all array combinations
        bins -- dictionary result from create_bins
        binned_probabilities -- list of dictionaries, mof array, gas, pmfs
    """

    array_kld_results = []
    for array in list_of_arrays:
        dict_temp = {'MOF_Array' : array}
        array_name = '%s' % ' '.join(array)

        # Calculate Aboslute KLD
        pmfs_per_array_abs = [row[array_name] for row in all_array_pmf_results]
        reference_prob_abs = 1/len(pmfs_per_array_abs)
        abs_kld = sum([float(pmf)*log(float(pmf)/reference_prob_abs,2) for pmf in pmfs_per_array_abs if pmf != 0])
        dict_temp.update({'Absolute_KLD' : round(abs_kld,4)})

        # Calculate Component KLD
        for gas in gases:
            reference_prob_comp = 1/len(bins)
            pmfs_per_array_comp = [row[array_name] for row in binned_probabilities if '%s bin' % gas in row.keys()]
            kld_comp = sum([float(pmf)*log(float(pmf)/reference_prob_comp,2) for pmf in pmfs_per_array_comp if pmf != 0])
            dict_temp.update({'%s KLD' % gas : round(kld_comp,4)})

        # Calculate Joint KLD
        product_temp = reduce(operator.mul, [dict_temp['%s KLD' % gas] for gas in gases], 1)
        dict_temp.update({'Joint_KLD' : product_temp})
        dict_temp.update({'Array_Size' : len(dict_temp['MOF_Array'])})
        array_kld_results.append(dict_temp)

    return array_kld_results


def choose_arrays(gases, num_mofs, array_kld_results, num_best_worst):
    """
    ----- Rank MOF arrays by KLD -----
    Keyword arguments:
        gases -- list of gases
        num_mofs -- minimum and maximum number of mofs in an array, usr specified in config file
        num_best_worst - number of the best and worst mofs of eash array size to save
        array_kld_results -- list of dictionaries including, mof array, gas, and corresponding kld
    """

    # Saves best and worst arrays of each array size, ranking by absolute KLD
    best_ranked_by_abskld = sorted(array_kld_results, key=lambda k: k['Absolute_KLD'], reverse=True)
    worst_ranked_by_abskld = sorted(array_kld_results, key=lambda k: k['Absolute_KLD'])
    array_list_abskld = [best_ranked_by_abskld, worst_ranked_by_abskld]
    best_and_worst_arrays_by_absKLD = []
    for ranked_list in array_list_abskld:
        for array_size in range(min(num_mofs),max(num_mofs)+1):
            index = 0
            for array in ranked_list:
                if index < num_best_worst and len(array['MOF_Array']) == array_size:
                    best_and_worst_arrays_by_absKLD.append(array)
                    index += 1

    # Saves best and worst arrays of each array size, ranking by joint KLD
    best_ranked_by_jointkld = sorted(array_kld_results, key=lambda k: k['Joint_KLD'], reverse=True)
    worst_ranked_by_jointkld = sorted(array_kld_results, key=lambda k: k['Joint_KLD'])
    array_list_jointkld = [best_ranked_by_jointkld, worst_ranked_by_jointkld]
    best_and_worst_arrays_by_jointKLD = []
    for ranked_list in array_list_jointkld:
        for array_size in range(min(num_mofs),max(num_mofs)+1):
            index = 0
            for array in ranked_list:
                if index < num_best_worst and len(array['MOF_Array']) == array_size:
                    best_and_worst_arrays_by_jointKLD.append(array)
                    index += 1

    # Saves best and worst arrays of each size, ranking by gas KLD
    best_and_worst_arrays_by_gasKLD = []
    for gas in gases:
        best_ranked_per_gas  = sorted(array_kld_results, key=lambda k: k['%s KLD' % gas], reverse=True)
        worst_ranked_per_gas = sorted(array_kld_results, key=lambda k: k['%s KLD' % gas])
        array_list_gaskld = [best_ranked_per_gas, worst_ranked_per_gas]
        for array_size in range(min(num_mofs),max(num_mofs)+1):
            for ranked_list in array_list_gaskld:
                index = 0
                for array in ranked_list:
                    if index < num_best_worst and len(array['MOF_Array']) == array_size:
                        best_and_worst_arrays_by_gasKLD.append(array)
                        index +=1

    return best_and_worst_arrays_by_absKLD, best_and_worst_arrays_by_jointKLD, best_and_worst_arrays_by_gasKLD

def assign_array_ids(list_of_arrays):
    """
    Assign numbers to each array for shorthand notation
    Can probably be written better to make enumerating less dependent on how the passed in list is ordered.
    Will save this for suture updates, since it currently works without issue.
    """
    array_id_dict = {}
    i = 0
    num_elements = 0
    for array in list_of_arrays:
        if num_elements == len(array):
            i += 1
            num_elements = num_elements
        else:
            i = 1
            num_elements = len(array)
        array_id = str(num_elements)+'-'+str(i)
        array_name = '%s' % ' '.join(array)
        array_id_dict.update({array_name: array_id})

    filename = 'saved_array_pmfs_unbinned/%s/array_id_list.csv' % (timestamp)
    with open(filename,'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        for key, val in array_id_dict.items():
            writer.writerow([key, val])

    return(array_id_dict)

def save_element_pmf_data(element_pmf_results, stdev, mrange):
    """
    ----- Saves pmf and mole fraction data for each gas/MOF combination -----
    Keyword arguments:
        element_pmf_results -- list of dictionaries with all pmf values
    """

    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    data_frame = pd.DataFrame(element_pmf_results)
    data_frame.to_csv('saved_element_pmfs/%s_stdev_%s_mrange_%s.csv' % (stdev, mrange, timestamp), sep='\t')


def save_unbinned_array_pmf_data(gases, list_of_arrays, list_of_array_ids, all_array_pmf_results):
    """
    ----- Saves pmf and mole fraction data for each gas/MOF array combination -----
    Keyword arguments:
        gases -- list of gases specified by user
        list_of_arrays -- list of all arrays, and MOFs in each array
        bins -- dictionary result from create_bins
        binned_probabilities -- list of dictionaries, mof array, gas, pmfs
    """

    # Make directory to store pmf data
    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    os.makedirs("saved_array_pmfs_unbinned/%s" % timestamp)

    # Generate array data and write to file
    header = []
    for gas in gases:
        header.extend([str(gas)])
    header.extend(['PMF'])

    comps_to_save = []
    comps_to_save_alt =[]
    for row in all_array_pmf_results:
        comps_to_save.append([{'%s' % gas: row[gas]} for gas in gases])
        comps_to_save_alt.append(np.transpose([row[gas] for gas in gases]))

    for array in list_of_arrays:
        array_name = '%s' % ' '.join(array)
        pmfs_to_save = [{'PMF': row[array_name]} for row in all_array_pmf_results]
        pmfs_to_save_alt = [[row[array_name]] for row in all_array_pmf_results]
        filename = "saved_array_pmfs_unbinned/%s/%s.csv" % (timestamp, str(list_of_array_ids[array_name]))
        pmf_data = np.column_stack((comps_to_save, pmfs_to_save))

        # Using Alt form of data
        pmf_data_alt = np.column_stack((comps_to_save_alt, pmfs_to_save_alt))
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile,delimiter='\t')
            writer.writerow(header)
            for line in pmf_data_alt:
                writer.writerow(line)


def save_binned_array_pmf_data(gases, list_of_arrays, list_of_array_ids, bins, binned_probabilities):
    """
    ----- Saves pmf and mole fraction data for each gas/MOF array combination -----
    Keyword arguments:
        gases -- list of gases specified by user
        list_of_arrays -- list of all arrays, and MOFs in each array
        bins -- dictionary result from create_bins
        binned_probabilities -- list of dictionaries, mof array, gas, pmfs
    """

    # Make directory to store pmf data
    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    os.makedirs("saved_array_pmfs_binned/%s" % timestamp)

    # Generate array data and write to file
    for gas in gases:
        comps_to_save = [bin[gas] for bin in bins][0:len(bins)-1]
        for array in list_of_arrays:
            array_name = '%s' % ' '.join(array)
            pmfs_to_save = [row[array_name] for row in binned_probabilities if '%s bin' % gas in row.keys()]
            pdfs_to_save = len(comps_to_save) * np.array(pmfs_to_save)
            filename = "saved_array_pmfs_binned/%s/%s_%s.csv" % (timestamp, str(list_of_array_ids[array_name]), str(gas))
            pdf_data = np.column_stack((comps_to_save, pdfs_to_save))
            with open(filename,'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter="\t")
                for line in pdf_data:
                    writer.writerow(line)


def plot_binned_array_pmf_data(gases, list_of_arrays, list_of_array_ids, bins, binned_probabilities):
    """
    ----- Plots pmf vs mole fraction for each gas/MOF array combination -----
    Keyword arguments:
        gases -- list of gases specified by user
        list_of_arrays -- list of all array combinations
        bins -- dictionary result from create_bins
        binned_probabilities -- list of dictionaries, mof array, gas, pmfs
    """

    # Make directory to store figures
    timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    os.makedirs('saved_array_pmfs_binned_figures/%s' % timestamp)

    # Generate the plots
    array_names = ['%s' % ' '.join(array) for array in list_of_arrays]
    for gas in gases:
        for array in array_names:
            # X-axis, list of mole fracs to plot, for relevant gas
            comps_to_plot = [bin[gas] for bin in bins][0:len(bins)-1]
            # Y-axis, list of pmf values to plot
            pmfs_to_plot = [row[array] for row in binned_probabilities if '%s bin' % gas in row.keys()]
            pdfs_to_plot = len(comps_to_plot) * np.array(pmfs_to_plot)
            # Plot and save figure in a directory 'figures'
            plot_PMF = plt.figure()
            plt.rc('xtick', labelsize=20)
            plt.rc('ytick', labelsize=20)
            plt.plot(comps_to_plot, pdfs_to_plot, 'ro-')
            plt.title('Array: %s, Gas: %s' % (list_of_array_ids[array], gas))
            plt.savefig("saved_array_pmfs_binned_figures/%s/%s_%s.png" % (timestamp, 'Array#'+list_of_array_ids[array]+'_(See Key)', str(gas)))
            plt.close(plot_PMF)
