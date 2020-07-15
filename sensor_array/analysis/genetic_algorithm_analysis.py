"""
Original Code by: Jenna Gustafson
Modifications by: Brian Day

Analysis of the arrays mimics the process_mass_data.py code, but rather than
brute force generating and analyzing all arrays, an algorithm is used to attempt
to find the best array of many posibilities. Currently (i.e. once operational)
this code will be used seprately, but for the same purpose as the
process_mass_data.py code, but for array sizes with too many possibilities
to analyze by brute force.

For the sake of consistency in the approch, these two codes should be merged to
some degree in the future, so that identical functions are being called. The
number of possible arrays can act as a switch for the two methods (i.e. brute
force or genetic algorithm).
"""

# --------------------------------------------------
# ----- Import Python Packages ---------------------
# --------------------------------------------------
import copy
import csv
import functools
import numpy as np
import operator
import os
import pandas as pd
import random
import scipy.interpolate as si
import scipy.stats as ss
import sys
import time
import yaml

from datetime import datetime
from functools import reduce
from itertools import combinations
from math import isnan, log
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import spline

# --------------------------------------------------
# ----- Functions from process_mass_data.py --------
# Try not to change these here! This will make recombining with other files
# easier later.
# --------------------------------------------------
def read_data_as_dict(filename):
    with open(filename,newline='') as csvfile:
        output_data = csv.DictReader(csvfile, delimiter="\t")
        return list(output_data)

def write_data_as_tabcsv(filename, data):
    with open(filename,'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        for line in data:
            writer.writerow([line])
    return(writer)

def write_GA_results_messy(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerows(data)
    return(writer)

def read_GA_results_messy(filename):
    import ast
    with open(filename, newline='') as csvfile:
        output_data = csv.reader(csvfile, delimiter="\t")
        output_data = list(output_data)
        for i in range(len(output_data)):
            for j in range(len(output_data[i])):
                output_data[i][j] = ast.literal_eval(output_data[i][j])
        return output_data

def write_GA_results_clean(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        for i in range(len(data)):
            for j in range(len(data[i])):
                writer.writerow([data[i][j]])
                if j == len(data[i])-1 and i != len(data):
                    writer.writerow('-')
    return(writer)

def read_GA_results_clean(filename):
    import ast
    with open(filename, newline='') as csvfile:
        output_data = csv.reader(csvfile, delimiter="\t")
        output_data = list(output_data)
        full_array = []
        start = 0
        for i in range(len(output_data)):
            row = output_data[i][0]
            if row == '-':
                end = i
                temp_array = []
                for j in range(start,end):
                    temp_row = ast.literal_eval(output_data[j][0])
                    temp_array.append(temp_row)
                full_array.append(temp_array)
                start = i+1
        return full_array

def plot_GA_results(filename, data, num_generations, array_size, seek='best', seek_by='Absolute_KLD'):
    best_kld_each_gen = []
    worst_kld_each_gen = []
    all_kld_boxplot = []
    generations = []
    num_generations_total = np.sum(num_generations)
    for j in range(num_generations_total):
        worst_kld_temp = data[j][-1][seek_by]
        worst_kld_each_gen.append(worst_kld_temp)
        best_kld_temp = data[j][0][seek_by]
        best_kld_each_gen.append(best_kld_temp)
        all_kld_temp = []
        for k in range(len(data[j])):
            kld = data[j][k][seek_by]
            all_kld_temp.append(kld)
            all_kld_boxplot.append({'y':np.array(all_kld_temp), 'type':'box', 'marker':{'color': 'hsl(180.0,50%,50%)'}, 'name':j+1})
        current_gen = data[j][0]['Generation']
        generations.append(current_gen)

    plt.clf()

    if seek == 'best':
        plt.plot(generations, best_kld_each_gen, 'o-', color='#1f77b4', alpha=1.0)
        plt.plot(generations, worst_kld_each_gen, 'o-', color='#ff7f0e', alpha=0.5)
        plt.legend(['Best KLD','Worst KLD'], loc='upper left')
    elif seek == 'worst':
        plt.plot(generations, best_kld_each_gen, 'o-', color='#ff7f0e', alpha=1.0)
        plt.plot(generations, worst_kld_each_gen, 'o-', color='#1f77b4', alpha=0.5)
        plt.legend(['Worst KLD','Best KLD'], loc='upper left')

    for i in range(1,len(num_generations)):
        plt.axvline(x=np.sum(num_generations[0:i])+0.5, c='black', ls='dashed', lw=1)

    plt.xlabel('Generation', Fontsize=16)
    plt.xlim(0,num_generations_total)
    plt.xticks(np.linspace(0,num_generations_total,11))

    plt.ylabel(seek_by.replace('_', ' ')+' Value', Fontsize=16)
    plt.ylim(0,10)
    plt.yticks(np.linspace(0,10,11))

    plt.title('Genetic Algorithm Approach \n Array Size = %s' % array_size, Fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)

def yaml_loader(filepath):
    with open(filepath, 'r') as yaml_file:
        data = yaml.load(yaml_file)
    return(data)

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
    return(exp_results_full, exp_results_mass, exp_mof_list)

def import_simulated_data(sim_results_import, mof_list, mof_densities, gases):
    sim_results_full = []
    for mof in mof_list:
        for row in sim_results_import:
            if row['MOF'] == mof:
                mass = float(mof_densities[mof]) * float(row['Mass'])
                sim_results_temp = row.copy()
                sim_results_temp.update({'Mass_mg/cm3' : mass})
                sim_results_full.extend([sim_results_temp])
    return(sim_results_full)

def calculate_element_pmf(exp_results_full, sim_results_full, mof_list, stdev, mrange):

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
    """
    Function used for 'manually' adding random error to smoothed data. Regardless of the choosen
    error amount, the lower bound on the adsorbed mass is still 0 (i.e. no negative adsorbed masses).
    """
    random.seed(seed)
    for row in sim_results_import:
        if row['Mass_mg/cm3']-error <= 0:
            row['Mass_mg/cm3'] += random.uniform(-row['Mass_mg/cm3'],error)
        else:
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

    return(element_pmf_results)

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
    return(single_array_pmf_results)

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

    return(bins)

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
                     row.update({'%s bin' % gas : lower_bin})

        # Loops through all of the bins and takes sum over all pmfs in that bin.
        all_bins_temp_sum = []
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
            with_sum = {'%s bin' % gas : bin[gas]} #copy.deepcopy(single_bin_temp)
            for array in array_names:
                if array_pmfs_temp[array] == []:
                    with_sum.update({array : 0})
                else:
                    with_sum.update({array : sum(array_pmfs_temp[array])})
            all_bins_temp_sum.append(with_sum)

        # Creates list of binned probabilities, already normalized
        binned_probabilities_sum.extend(all_bins_temp_sum)

    return(binned_probabilities_sum)

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

    return(array_kld_results)

# --------------------------------------------------
# ----- Functions for Algorithm --------------------
# --------------------------------------------------
def create_random_array(array_size, mof_list):
    array = random.sample(mof_list, array_size)
    array.sort()
    return(array)

def create_first_generation(population_size, array_size, mofs_list):
    population = []
    for i in range(population_size):
        new_array = create_random_array(array_size,mofs_list)
        population.append(new_array)
    return(population)

def remove_duplicate_arrays(array_list, mofs_list):
    """
    Removes any duplicate arrays from a given generation and replaces them with randomly generated
    arrays. Since the elitisim strategy guarantees that all of the parents are cloned in the new
    generation, this strategy should be fine.
    """
    orig_len = len(array_list)
    array_size = len(array_list[0])

    array_list_temp = [tuple(x) for x in array_list]
    array_list = [list(x) for x in set(array_list_temp)]
    current_len = len(array_list)

    # Repopulate up to original size, checking for duplicates along the way
    while current_len < orig_len:
        rand_array = create_random_array(array_size, mofs_list)
        array_list.append(rand_array)
        array_list_temp = [tuple(x) for x in array_list]
        array_list = [list(x) for x in set(array_list_temp)]
        current_len = len(array_list)

    return(array_list)

def calculate_all_array_pmf(array_list, gases, element_pmf_results, comp_set_dict):

    all_array_pmf_results = []
    for i in range(len(array_list)):
        array = array_list[i]
        array_name = '%s' % ' '.join(array)
        single_array_pmf_results = calculate_array_pmf(array, element_pmf_results)
        if array == array_list[0]:
            # First Array: Set up dictionary with keys
            for index in range(len(comp_set_dict)):
                array_dict = { array_name : single_array_pmf_results[index]}
                for gas in gases:
                    array_dict.update({ '%s' % gas : float(comp_set_dict[index][gas])})
                all_array_pmf_results.extend([array_dict])
        else:
            # Not First Array: Update Dictionary
            for index in range(len(comp_set_dict)):
                # all_array_pmf_results[index]['%s' % ' '.join(array)] = single_array_pmf_results[index]
                all_array_pmf_results[index].update({array_name : single_array_pmf_results[index]})
    return(all_array_pmf_results)

def calculate_all_array_kld(num_bins, mof_list, array_list, gases, element_pmf_results, all_array_pmf_results):
    bins = create_bins(gases, num_bins, mof_list, element_pmf_results)
    binned_probabilities_sum = bin_compositions(gases, bins, array_list, all_array_pmf_results)
    all_array_kld_results = calculate_kld(gases, array_list, bins, all_array_pmf_results, binned_probabilities_sum)
    return(all_array_kld_results)

def sort_population(population_fitness, seek='best', seek_by='Absolute_KLD'):
    if seek == 'best':
        ordered_population = sorted(population_fitness, key=lambda k: k[seek_by], reverse=True)
    elif seek == 'worst':
        ordered_population = sorted(population_fitness, key=lambda k: k[seek_by], reverse=False)
    return(ordered_population)

def choose_parents(population_sorted, num_best, num_lucky):
    result = []
    result.extend(population_sorted[0:num_best])
    result.extend(random.sample(population_sorted[num_best:], num_lucky))
    return(result)

def create_child_by_crossover(parent1, parent2):
    """
    Combine all parent material into a single pool, removing duplicates by temporarily creating a set.
    Create a child from the pool of parent material.
    """
    parent_material = list(set(parent1+parent2))
    child = random.sample(parent_material, len(parent1))
    child.sort()
    return(child)

def create_children_by_crossover(parents, num_children):
    """
    Create a set of children from a given set of parents, but always retaining a clone of each parent
    in the set of children (i.e. the elitism strategy).
    """
    children = []
    num_parents = len(parents)
    for i in range(num_parents):
        children.append(parents[i])
    for i in range(num_parents, num_children):
        parent1, parent2 = random.sample(parents,2)
        child = create_child_by_crossover(parent1, parent2)
        children.append(child)
    return(children)

def create_child_by_mutation(parent, mutation_rate, mofs_list):
    child = copy.deepcopy(parent)

    # Create a list of all mofs not currently in the array
    mof_list_temp_orig = [mof for mof in mofs_list if mof not in child]
    mof_list_temp_new = mof_list_temp_orig

    for i in range(len(child)):
        if random.random() < mutation_rate:
            child[i] = random.choice(mof_list_temp_new)
            # Remove newly added mof from the list of mofs not in the array (probably a faster way to do this than as below)
            mof_list_temp_new = [mof for mof in mof_list_temp_orig if mof not in child]
            # If all mofs not originally in the array are added, make a list of mofs not in the current array state
            # 'if not x:' checks to see if x is None
            if not mof_list_temp_new:
                mof_list_temp_new = [mof for mof in mofs_list if mof not in child]

    child.sort()
    return(child)

def create_children_by_mutation(parents, num_children, mutation_rate, mofs_list):
    children = []
    num_parents = len(parents)
    for i in range(num_parents):
        children.append(parents[i])
    for i in range(num_parents, num_children):
        parent = random.choice(parents)
        child = create_child_by_mutation(parent, mutation_rate, mofs_list)
        children.append(child)
    return(children)

# --------------------------------------------------
# ----- Genetic Algorithm Function / Call ----------
# --------------------------------------------------
def run_genetic_algorithm(first_gen, generation_start_num, array_size, mofs_list, num_best, num_lucky, population_size, \
    num_generations, mutation_rate, element_pmf_results, gases, num_bins, seek='best', seek_by='Absolute_KLD'):

    # Save list of dictionaries for first MOF for use as a list of all gas mole fractions
    comp_set_dict = [row for row in element_pmf_results if row['MOF'] == mofs_list[0]]

    # Analyze first generation, generate and analyze subsequent generations
    all_arrays_list_by_generation = []
    all_array_results_by_generation = []

    # generation = create_first_generation(population_size,array_size,mofs_list)
    generation = remove_duplicate_arrays(first_gen, mofs_list)

    for i in range(num_generations):
        all_arrays_list_by_generation.append(generation)
        genx_pmf_results = calculate_all_array_pmf(generation, gases, element_pmf_results, comp_set_dict)
        genx_kld_results = calculate_all_array_kld(num_bins, mofs_list, generation, gases, \
            element_pmf_results, genx_pmf_results)
        genx_kld_results_sorted = sort_population(genx_kld_results, seek=seek, seek_by=seek_by)
        genx_list_sorted = [x['MOF_Array'] for x in genx_kld_results_sorted]

        for row in genx_kld_results_sorted:
            row['Generation'] = generation_start_num+i

        all_array_results_by_generation.append(genx_kld_results_sorted)

        parents = choose_parents(genx_list_sorted, num_best, num_lucky)
        children = []
        # children.extend(create_children_by_crossover(parents, population_size))
        children.extend(create_children_by_mutation(parents, population_size, mutation_rate, mofs_list))
        children = remove_duplicate_arrays(children, mofs_list)
        generation = children

    return(all_arrays_list_by_generation, all_array_results_by_generation)
