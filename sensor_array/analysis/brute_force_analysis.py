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
import operator
import os
import sys
from datetime import datetime
from functools import reduce
from itertools import combinations
from math import isnan, log
import random

import numpy as np
import pandas as pd
import scipy.interpolate as si
import scipy.stats as ss
import yaml
import matplotlib
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import spline
import ternary

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
    Can probably omit exp_results_mass in the future and just use the full
    dictionary for further analysis.
    """
    exp_results_full = []
    exp_results_mass = []
    exp_mof_list = []
    for mof in mof_list:
        for row in exp_results_import:
            if row['MOF'] == mof:
                mass = float(mof_densities[mof]) * float(row['Mass'])
                row['Mass_mg/cm3'] = mass
                exp_results_full.extend([row])
                exp_results_mass.append({'MOF' : mof, 'Mass' : mass})
                exp_mof_list.append(mof)
            else:
                None
    return exp_results_full, exp_results_mass, exp_mof_list


def import_simulated_data(sim_results_import, mof_list, mof_densities, gases):
    sim_results_full = []
    for mof in mof_list:
        for row in sim_results_import:
            if row['MOF'] == mof:
                mass = float(mof_densities[mof]) * float(row['Mass'])
                row['Mass_mg/cm3'] = mass
                sim_results_full.extend([row])
    return sim_results_full


def create_comp_list(simulated_data, mof_list, gases):
    # Extract a list of all possible compositions
    comp_list = []
    for row in simulated_data:
        if row['MOF'] == mof_list[0]:
            temp_dict = {}
            for gas in gases:
                temp_dict[gas] = float(row[gas])
            comp_list.extend([temp_dict])

    # Extracta  a list of all possible mol fractions for each individual gas
    mole_fractions = {}
    for gas in gases:
        temp_list = []
        for row in comp_list:
            temp_list.extend([float(row[gas])])
        temp_list_no_dups = list(set(temp_list))
        temp_list_no_dups.sort()
        mole_fractions[gas] = temp_list_no_dups
    return comp_list, mole_fractions


# Set of potential smoothing functions.
def moving_average_smooth(sim_results_import, mof_list, gases, comp_list, mole_fractions, num_points=1):
    """
    Smooth a set of simulated data by averageing the data points around a given value. Central points
    sample from points in all directions. Boundary points sample from points along the boundary.
    Corner points remain unchanged.
    With current data strucutre, this function will be highly inefficient. May not be slow enough to
    warrant changing, but worth noting.
    """

    # Create a temporary data set for the mof of interest. Consider doing this in a separate function.
    data_smoothed = []
    for mof in mof_list:

        temp_data = []
        for row in sim_results_import:
            if row['MOF'] == mof:
                temp_data.extend([row])

        for comp_main in comp_list:
            temp_dict = {}
            for row in temp_data:
                if [float(row[gas]) for gas in gases] == [comp_main[gas] for gas in gases]:
                    temp_dict = dict(row)
            for gas in gases:
                temp_dict[gas] = comp_main[gas]

            # Create list of possible compoenent mole fractions to use in determining neighbors.
            allowed_comps = {}
            for gas in gases:
                index = mole_fractions[gas].index(comp_main[gas])
                for i in range(0, num_points+1):
                    if index-i >= 0 and index+i <= len(mole_fractions[gas])-1:
                        allowed_comps[gas] = [mole_fractions[gas][i] for i in range(index-i, index+i+1)]

            # Extract a list of neighbors (include main point) based on the allowed composititons.
            # Since this includes the main point, len(temp_data_subset) will always >= 1.
            temp_data_subset = []
            for row in temp_data:
                boolean_array = []
                for gas in gases:
                    if allowed_comps.get(gas) != None:
                        if float(row[gas]) in allowed_comps[gas]:
                            boolean_array.extend([1])
                        else:
                            boolean_array.extend([0])
                    else:
                        boolean_array.extend([0])
                boolean_array = [i for i in boolean_array if i == 0]
                if len(boolean_array) == 0:
                    temp_data_subset.extend([row])

            # Calculate the average mass of all points in the subset.
            if len(temp_data_subset) != 0:
                mass_total = 0.0
                for row in temp_data_subset:
                    mass_total += float(row['Mass_mg/cm3'])
                mass_average = mass_total / len(temp_data_subset)
                temp_dict['Mass_mg/cm3'] = mass_average
                temp_dict['Num_points'] = len(temp_data_subset)
            else:
                mass_average = 0.0
                temp_dict['Mass_mg/cm3'] = temp_dict['Mass_mg/cm3']
                temp_dict['Num_points'] = len(temp_data_subset)

            data_smoothed.extend([temp_dict])

    return data_smoothed


def reintroduce_random_error(sim_results_import, error=1, seed=0):
    random.seed(seed)
    for row in sim_results_import:
        row['Mass_mg/cm3'] += random.uniform(-error,error)

    return sim_results_import


def convert_experimental_data(exp_results_import, sim_results_import, mof_list, gases):
    """
    This is a function only meant to convert the set of experimental data from the unsmoothed set to
    the smoothed set, without the need for additional work outside of the code. Should probably streamline
    this process better in the future.
    """
    exp_comp = {gas: exp_results_import[0][gas] for gas in gases}
    for row_exp in exp_results_import:
        for row_sim in sim_results_import:
            if row_sim['MOF'] == row_exp['MOF'] and {gas: float(row_sim[gas]) for gas in gases} == {gas: float(row_exp[gas]) for gas in gases}:
                row_exp['Mass_mg/cm3'] = row_sim['Mass_mg/cm3']

    return exp_results_import

# def polyfit_average_smooth(sim_results_import):
# 	"""
# 	Custom smoothing strategy in which a polynomial is fit along a set of points along each axis. The
# 	new value of the averaged point is the average of each of the polynomial functions evaluated at
# 	the given point.
# 	"""
# 	for mof in mof_list:
# 		for gas in gases:
# 			for comp in comp_list:
# 	return data_smoothed
#
#
# def kernal_smooth(sim_results_import):
# 	return data_smoothed
#
#
# def spline_smooth(sim_results_import):
# 	return data_smoothed


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


def calculate_element_pmf(exp_results_full, sim_results_full, mof_list, stdev, mrange, type='mass'):
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
            probs_exact = []
            if type == 'mass':
                a, b = 0, 2*float(max(all_masses_sim))
                mu, sigma = float(mass_exp), float(stdev)
                alpha, beta = ((a-mu)/sigma), ((b-mu)/sigma)
            if type == 'percent':
                a, b = 0, float(max(all_masses_sim)) * (1 + mrange)
                mu, sigma = float(mass_exp), float(stdev)*float(mass_exp)
                alpha, beta = ((a-mu)/sigma), ((b-mu)/sigma)
            for mass_sim in all_masses_sim:
                upper_prob = ss.truncnorm.cdf(float(mass_sim) * (1 + mrange), alpha, beta, loc = mu, scale = sigma)
                lower_prob = ss.truncnorm.cdf(float(mass_sim) * (1 - mrange), alpha, beta, loc = mu, scale = sigma)
                probs_range.append(upper_prob - lower_prob)
                prob_singlepoint = ss.truncnorm.pdf(float(mass_sim), alpha, beta, loc=mu, scale=sigma)
                probs_exact.append(prob_singlepoint)
            sum_probs_range = sum(probs_range)
            norm_probs_range = [(i/sum_probs_range) for i in probs_range]
            sum_probs_exact = sum(probs_exact)
            norm_probs_exact = [(i/sum_probs_exact) for i in probs_exact]

            # Update dictionary with pmf for each MOF
            new_temp_dict = []
            # Initialize the dictioary
            if mof_temp_dict == []:
                for index in range(len(norm_probs_range)):
                    mof_temp_dict = all_results_sim[index].copy()
                    mof_temp_dict['PMF_Range'] = norm_probs_range[index]
                    mof_temp_dict['PMF_Exact'] = norm_probs_exact[index]
                    new_temp_dict.extend([mof_temp_dict])
                new_temp_dict_2 = new_temp_dict
            # Add to the exisitng dictionary
            else:
                for index in range(len(norm_probs_range)):
                    mof_temp_dict = new_temp_dict_2[index].copy()
                    mof_temp_dict['PMF_Range'] = norm_probs_range[index]
                    mof_temp_dict['PMF_Exact'] = norm_probs_exact[index]
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


def calculate_all_arrays_list(mof_list, num_mofs):
    mof_array_list = []
    array_size = min(num_mofs)
    while array_size <= max(num_mofs):
        mof_array_list.extend(list(combinations(mof_list, array_size)))
        array_size += 1

    return mof_array_list


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
                array_dict = {' '.join(mof_array) : single_array_pmf_results[index]}
                for gas in gases:
                    array_dict[gas] = float(comp_set_dict[index][gas])
                all_array_pmf_results.extend([array_dict])
        else:
            # Not First Array: Update Dictionary
            for index in range(len(comp_set_dict)):
                all_array_pmf_results[index][' '.join(mof_array)] = single_array_pmf_results[index]

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
        bin_points.append(np.linspace(lower_lim, upper_lim, num=num_bins+1, endpoint=True))
    bin_points = np.transpose(np.vstack(bin_points))

    # Reformat bin_points
    bins = []
    for row in bin_points:
        bins.append({gases[i] : row[i] for i in range(len(gases))})

    return bins


def create_comp_set_dict(element_pmf_results, mof_list):
    comp_set_dict = [row for row in element_pmf_results if row['MOF'] == mof_list[0]]
    return comp_set_dict


def bin_compositions_single_array(gases, bins, array, single_array_pmf_results, comp_set_dict):
    # Create a dictionary from array_pmf_results
    array_key = ' '.join(array)

    array_dict = []
    for index in range(len(comp_set_dict)):
        array_dict_temp = {array_key : single_array_pmf_results[index]}
        for gas in gases:
            array_dict_temp[gas] = float(comp_set_dict[index][gas])
        array_dict.append(array_dict_temp)

    # Loop through dictionary and assign bins
    for gas in gases:
        for row in array_dict:
            for i in range(1,len(bins)):
                lower_bin = bins[i-1][gas]
                upper_bin = bins[i][gas]
                gas_comp = float(row[gas])
                if gas_comp >= lower_bin and gas_comp < upper_bin:
                    row['%s bin' % gas] = lower_bin

    # Loops through all of the bins and takes sum over all pmfs in that bin.
    binned_probabilities_sum = []
    for gas in gases:
        all_bins_temp_sum = []
        for bin in bins[0:len(bins)-1]:
            pmfs_temp = []
            for row in array_dict:
                if bin[gas] == row['%s bin' % gas]:
                    pmfs_temp.append(row[array_key])

            single_bin_temp = {'%s bin' % gas : bin[gas]}
            with_sum = copy.deepcopy(single_bin_temp)
            if pmfs_temp == []:
                with_sum[array_key] = 0
            else:
                with_sum[array_key] = sum(pmfs_temp)
            all_bins_temp_sum.append(with_sum)

        # Creates list of binned probabilities, already normalized
        binned_probabilities_sum.extend(all_bins_temp_sum)

    return binned_probabilities_sum, array_dict


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
                     row['%s bin' % gas] = lower_bin

        # Loops through all of the bins and takes sum over all pmfs in that bin.
        all_bins_temp_sum = []
        all_bins_temp_max = []
        array_names = [' '.join(array) for array in list_of_arrays]
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
                    with_sum[array] = 0
                    with_max[array] = 0
                else:
                    with_sum[array] = sum(array_pmfs_temp[array])
                    with_max[array] = max(array_pmfs_temp[array])
            all_bins_temp_sum.append(with_sum)
            all_bins_temp_max.append(with_max)

        # Creates list of binned probabilities, already normalized
        binned_probabilities_sum.extend(all_bins_temp_sum)
        binned_probabilities_max.extend(all_bins_temp_max)

    return binned_probabilities_sum, binned_probabilities_max


def calculate_single_array_kld(gases, array, bins, single_array_pmf_results, binned_probabilities):
    dict_temp = {'MOF_Array' : array}
    array_name = ' '.join(array)

    # Calculate Absolute KLD
    reference_prob_abs = 1/len(single_array_pmf_results)
    abs_kld = sum([float(pmf)*log(float(pmf)/reference_prob_abs,2) for pmf in single_array_pmf_results if pmf != 0])
    dict_temp['Absolute_KLD'] = round(abs_kld,4)

    # Calculate Component KLD
    for gas in gases:
        reference_prob_comp = 1/len(bins)
        pmfs_per_array_comp = [row[array_name] for row in binned_probabilities if '%s bin' % gas in row.keys()]
        kld_comp = sum([float(pmf)*log(float(pmf)/reference_prob_comp,2) for pmf in pmfs_per_array_comp if pmf != 0])
        dict_temp['%s KLD' % gas] = round(kld_comp,4)

    # Calculate Joint KLD
    product_temp = reduce(operator.mul, [dict_temp['%s KLD' % gas] for gas in gases], 1)
    dict_temp['Joint_KLD'] = product_temp
    dict_temp['Array_Size'] = len(dict_temp['MOF_Array'])

    return dict_temp


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
        array_name = ' '.join(array)

        # Calculate Absolute KLD
        pmfs_per_array_abs = [row[array_name] for row in all_array_pmf_results]
        reference_prob_abs = 1/len(pmfs_per_array_abs)
        abs_kld = sum([float(pmf)*log(float(pmf)/reference_prob_abs,2) for pmf in pmfs_per_array_abs if pmf != 0])
        dict_temp['Absolute_KLD'] = round(abs_kld,4)

        # Calculate Component KLD
        for gas in gases:
            reference_prob_comp = 1/len(bins)
            pmfs_per_array_comp = [row[array_name] for row in binned_probabilities if '%s bin' % gas in row.keys()]
            kld_comp = sum([float(pmf)*log(float(pmf)/reference_prob_comp,2) for pmf in pmfs_per_array_comp if pmf != 0])
            dict_temp['%s KLD' % gas] = round(kld_comp,4)

        # Calculate Joint KLD
        product_temp = reduce(operator.mul, [dict_temp['%s KLD' % gas] for gas in gases], 1)
        dict_temp['Joint_KLD'] = product_temp
        dict_temp['Array_Size'] = len(dict_temp['MOF_Array'])
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
    Will save this for future updates, since it currently works without issue.
    """
    array_id_dict = {}
    i = 0
    num_elements = 0
    for array in list_of_arrays:
        if num_elements == len(array):
            i += 1
        else:
            i = 1
            num_elements = len(array)
        array_id = str(num_elements)+'-'+str(i)
        array_name = ' '.join(array)
        array_id_dict[array_name] = array_id

    filename = 'array_id_list.csv'
    with open(filename,'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        for key, val in array_id_dict.items():
            writer.writerow([val, key])

    return array_id_dict


def save_element_pmf_data(element_pmf_results, stdev, mrange, timestamp):
    """
    ----- Saves pmf and mole fraction data for each gas/MOF combination -----
    Keyword arguments:
        element_pmf_results -- list of dictionaries with all pmf values
    """

    data_frame = pd.DataFrame(element_pmf_results)
    data_frame.to_csv('saved_element_pmfs/%s_stdev_%s_mrange_%s.csv' % (stdev, mrange, timestamp), sep='\t')


def save_unbinned_array_pmf_data(gases, list_of_arrays, list_of_array_ids, all_array_pmf_results, timestamp):
    """
    ----- Saves pmf and mole fraction data for each gas/MOF array combination -----
    Keyword arguments:
        gases -- list of gases specified by user
        list_of_arrays -- list of all arrays, and MOFs in each array
        bins -- dictionary result from create_bins
        binned_probabilities -- list of dictionaries, mof array, gas, pmfs
    """

    # Make directory to store pmf data
    os.makedirs("saved_array_pmfs_unbinned/%s" % timestamp)

    # Generate array data and write to file
    header = copy.deepcopy(gases)
    header.extend(['PMF'])

    comps_to_save = []
    comps_to_save_alt =[]
    for row in all_array_pmf_results:
        comps_to_save.append([{'%s' % gas: row[gas]} for gas in gases])
        comps_to_save_alt.append([row[gas] for gas in gases])
        # Alt form of compositons saves them in array rather than dictionary form. Order is thus
        # hard-coded, but its slightly easier to work with later for plotting purposes.

    for array in list_of_arrays:
        array_name = ' '.join(array)
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


def prepare_ternary_dict_rgba(a,b,c,z,vmin,vmax,cmap):
    """
    Prepare data for use with 'ternary'
    """

    color_norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    alpha_norm = matplotlib.colors.Normalize(vmin=vmin-(vmax-vmin), vmax=vmax)
    color_map = matplotlib.cm.get_cmap(cmap)

    data_dict_rgba = dict()
    data_dict_zval = dict()
    for i in range(len(a)):
        A_out = np.round(a[i]*100,3)
        B_out = np.round(b[i]*100,3)
        C_out = np.round(c[i]*100,3)
        alpha = 1
        rgba = color_map(color_norm(z[i]))[:3] + (alpha,)
        # data_dict[(bottom, right, left)] = ()
        data_dict_rgba[(A_out, B_out, C_out)] = (rgba)
        data_dict_zval[(A_out, B_out, C_out)] = z[i]

    return(data_dict_rgba, data_dict_zval)


def prepare_ternary_dict_rgba_rescaled(a,b,c,z,vmin,vmax,cmap):
    """
    Prepare rescaled data for use with 'ternary'
    """

    rescale_sub = max([min(a), min(b), min(c)])
    rescale_den = max([max(a)-min(a), max(b)-min(b), max(c)-min(c)])

    color_norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    color_map = matplotlib.cm.get_cmap(cmap)

    data_dict_rgba = dict()
    data_dict_zval = dict()
    for i in range(len(a)):
        atemp = (a[i]-min(a))/rescale_den
        btemp = (b[i]-min(b))/rescale_den
        ctemp = (c[i]-min(c))/rescale_den
        A_out = np.round(atemp*100,3)
        B_out = np.round(btemp*100,3)
        C_out = np.round(ctemp*100,3)
        rgba = color_map(color_norm(z[i]))
        # data_dict[(bottom, right, left)] = ()
        data_dict_rgba[(A_out, B_out, C_out)] = (rgba)
        data_dict_zval[(A_out, B_out, C_out)] = z[i]

    return(data_dict_rgba, data_dict_zval)


def harper_ternary(data_dict, z, array_id, vmin, vmax, cmap, use_rgba, polygon_sf):
    # Set image size and resolution
    matplotlib.rcParams['figure.dpi'] = 1200
    matplotlib.rcParams['figure.figsize'] = (3.25, 2.75)

    # Initialize the Plot
    scale = 100
    figure, tax = ternary.figure(scale=scale)
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    if use_rgba == True:
        tax.heatmap(data_dict, style="h", use_rgba=use_rgba, colorbar=False, polygon_sf=polygon_sf)
    elif use_rgba == False:
        tax.heatmap(data_dict, style="dt", use_rgba=use_rgba, polygon_sf=polygon_sf)

    # Set colorbar
    # vmin = min(z)
    # vmax = max(z)
    color_norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    color_map = matplotlib.cm.get_cmap(cmap)
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=color_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ticks=np.linspace(0,vmax,6), fraction=0.02, format='%.1f')
    cbar.ax.tick_params(labelsize=7)
    # cbar.ax.set_ylabel(r'$Mass\ [mg/cm^3]$', rotation=90, fontsize=6)
    cbar.ax.set_ylabel(r'Mass Uptake [$mg/cm^3$]', rotation=90, fontsize=9)

    # Draw boundary, gridlines, and ticks
    tax.boundary(linewidth=1.0)
    tax.gridlines(color=(0.5, 0.5, 0.5, 0.5), multiple=100/(6*4), linewidth=0.25)
    tax.gridlines(color=(0.5, 0.5, 0.5, 0.5), multiple=100/6, linewidth=0.5)
    tax.ticks(axis='b', ticks=[40,50,60,70,80,90,100], linewidth=1, multiple=100/7, tick_formats='%.0f', offset = 0.021, fontsize = 7)
    tax.ticks(axis='lr', ticks=[0,10,20,30,40,50,60], linewidth=1, multiple=100/7, tick_formats='%.0f', offset = 0.03, fontsize = 7)

    # Set axis labels and title
    title_fontsize = 9
    axis_fontsize = 9
    # tax.set_title("1 Mof Array: BISWEG", fontsize=title_fontsize, y=1.05)
    # tax.right_corner_label("X", fontsize=fontsize)
    # tax.top_corner_label("Y", fontsize=fontsize)
    # tax.left_corner_label("Z", fontsize=fontsize)
    tax.left_axis_label("Carbon Dioxide", fontsize=axis_fontsize, offset = 0.15)
    tax.right_axis_label("Oxygen", fontsize=axis_fontsize, offset = 0.15)
    tax.bottom_axis_label("Nitrogen", fontsize=axis_fontsize, offset = 0.05)

    plt.savefig('/Users/brian_day/Desktop/triplots/%s.png' % array_id, bbox_inches='tight')
    plt.close()
    return(figure)


def plot_element_mass_data(gases, mof_list, data, timestamp):
    cmap = 'viridis'
    psf = 1.05
    CO2_points = np.array([float(row['CO2']) for row in data if row['MOF'] == mof_list[0]])
    N2_points = np.array([float(row['N2']) for row in data if row['MOF'] == mof_list[0]])
    O2_points = np.array([float(row['O2']) for row in data if row['MOF'] == mof_list[0]])

    mass_values_minmax = []
    mass_values_minmax = [float(row['Mass_mg/cm3']) for row in data]
    # vmin = np.min(mass_values_minmax)
    # vmax = np.ceil(np.max(mass_values_minmax)*1000)/1000

    for mof in mof_list:
        mass_values = np.array([float(row['Mass_mg/cm3']) for row in data if row['MOF'] == mof])
        vmin = np.floor(np.min(mass_values)/10)*10
        vmax = np.ceil(np.max(mass_values)/10)*10

        (dict_rgba, dict_zval) = prepare_ternary_dict_rgba_rescaled(N2_points, O2_points, CO2_points, mass_values, vmin, vmax, cmap=cmap)
        figure = harper_ternary(dict_rgba, mass_values, mof, vmin, vmax, cmap=cmap, use_rgba=True, polygon_sf=psf)


def plot_unbinned_array_pmf_data(gases, list_of_arrays, list_of_array_ids, all_array_pmf_results, timestamp):
    cmap = 'viridis'
    psf = 1.05
    data = all_array_pmf_results
    CO2_points = np.array([float(row['CO2']) for row in data])
    N2_points = np.array([float(row['N2']) for row in data])
    O2_points = np.array([float(row['O2']) for row in data])

    PMF_values_minmax = []
    for row in all_array_pmf_results:
        for array in list_of_arrays:
            PMF_values_minmax.extend([float(row[array]) for row in data])
    vmin = np.min(PMF_values_minmax)
    # vmax = np.ceil(np.max(PMF_values_minmax)*1000)/1000
    vmax = 0.40

    for array in list_of_arrays:
        array_id = array
        PMF_values = []
        for row in all_array_pmf_results:
            PMF_values = np.array([float(row[array]) for row in data])

        (dict_rgba, dict_zval) = prepare_ternary_dict_rgba_rescaled(N2_points, O2_points, CO2_points, PMF_values, vmin, vmax, cmap=cmap)
        figure = harper_ternary(dict_rgba, PMF_values, array_id, vmin, vmax, cmap=cmap, use_rgba=True, polygon_sf=psf)


def save_binned_array_pmf_data(gases, list_of_arrays, list_of_array_ids, bins, binned_probabilities, timestamp):
    """
    ----- Saves pmf and mole fraction data for each gas/MOF array combination -----
    Keyword arguments:
        gases -- list of gases specified by user
        list_of_arrays -- list of all arrays, and MOFs in each array
        bins -- dictionary result from create_bins
        binned_probabilities -- list of dictionaries, mof array, gas, pmfs
    """

    # Make directory to store pmf data
    os.makedirs("saved_array_pmfs_binned/%s" % timestamp)

    # Generate array data and write to file
    for gas in gases:
        comps_to_save = [bin[gas] for bin in bins][0:len(bins)-1]
        for array in list_of_arrays:
            array_name = ' '.join(array)
            pmfs_to_save = [row[array_name] for row in binned_probabilities if '%s bin' % gas in row.keys()]
            pdfs_to_save = len(comps_to_save) * np.array(pmfs_to_save)
            filename = "saved_array_pmfs_binned/%s/%s_%s.csv" % (timestamp, str(list_of_array_ids[array_name]), gas)
            pdf_data = np.column_stack((comps_to_save, pdfs_to_save))
            with open(filename,'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter="\t")
                for line in pdf_data:
                    writer.writerow(line)


def plot_binned_array_pmf_data(gases, list_of_arrays, list_of_array_ids, bins, binned_probabilities, timestamp):
    """
    ----- Plots pmf vs mole fraction for each gas/MOF array combination -----
    Keyword arguments:
        gases -- list of gases specified by user
        list_of_arrays -- list of all array combinations
        bins -- dictionary result from create_bins
        binned_probabilities -- list of dictionaries, mof array, gas, pmfs
    """

    # Generate the plots
    # array_names = [' '.join(array) for array in list_of_arrays]
    # for array in array_names:
    array = list_of_arrays
    plt.figure(figsize=(3.5,2.5), dpi=600)
    plt.title('Binned Component\nProbabilities', fontsize=10)
    plt.xlim([0,1])
    plt.xticks(np.linspace(0,1,6), fontsize=8)
    plt.xlabel('Mole Fraction', fontsize=10)
    plt.ylim([0,0.7])
    plt.yticks(np.linspace(0,0.7,8), fontsize=8)
    plt.ylabel('Probability', fontsize=10)
    # plt.rc('xtick', labelsize=20)
    # plt.rc('ytick', labelsize=20)
    colors = ['red', 'green', 'blue']
    count = 0
    for gas in gases:
        # X-axis, list of mole fracs to plot, for relevant gas
        comps_to_plot = [bin[gas] for bin in bins][0:len(bins)-1]
        # Y-axis, list of pmf values to plot
        pmfs_to_plot = [row[array] for row in binned_probabilities if '%s bin' % gas in row.keys()]
        # pdfs_to_plot = len(comps_to_plot) * np.array(pmfs_to_plot)
        # Plot and save figure in a directory 'figures'
        plt.plot(comps_to_plot, pmfs_to_plot, 'o-', color=colors[count], markersize=3)
        count += 1

    plt.legend([r'$CO_2$',r'$N_2$', r'$N_2$'], fontsize=8)
    plt.tight_layout()
    plt.savefig("/Users/brian_day/Desktop/binned_pmf_figs/%s.png" % (array))
    plt.close()
