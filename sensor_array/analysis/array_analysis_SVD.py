"""
This will be used to design/analyze arrays where we leverage the Henry's Coefficients rather than the
GCMC simulations of the full composition space (impossible scale up).

This work closely follows the work presented by Sturluson et al. (Cory Simon's Group) in the paper
'Curating Metal Organic-Frameworks to Compose Robust Gas Sensor Arrays in Dilute Conditions' (2019).

The primary analysis method is the Singular Value Decomposition (SVD) mapping the composition space
to the sensor arrya response space.
"""

import ast
import csv
import itertools
import numpy as np
from numpy import linalg as nla
import re
import yaml

def yaml_loader(filepath):
    with open(filepath, 'r') as yaml_file:
        data = yaml.load(yaml_file)
    return(data)


def clean_MOF_name(MOF):
    MOF = re.sub(r'_eqeq', '', MOF)
    MOF = re.sub(r'_EQeq', '', MOF)
    MOF = re.sub(r'_EQEQ', '', MOF)
    MOF = re.sub(r'_v2', '', MOF)
    return MOF


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
            if type(temp_row['R^2']) == str or temp_row['R^2'] < 0:
                # Use if you don't want to include bad MOFs
                # continue
                # Use if you want place holder constants for non-fitted MOFs
                temp_row['Maximum Composition'] = 0.0
                temp_row['R^2'] = 'BAD'
                temp_row['k_H'] = 0.0
            temp_array.append(temp_row)
            full_array.extend(temp_array)
        return full_array


def unify_henrys_results():
    # (filepath, gases, filename, mof_list):
    filepath = '/Users/brian_day/Desktop/HC_work/HenrysConstants_Analysis_Results/'
    gases = ['acetone', 'ammonia', 'argon', 'CO2', 'hydrogen']
    filename = '_henrys_coefficients.csv'

    all_kH = {}
    for mof in mof_list:
        all_kH[mof] = {}

    for gas in gases:
        filepath_full = filepath+gas+'_AllRatios/'+filename
        data = read_kH_results(filepath_full)
        for row in data:
            row['MOF'] = clean_MOF_name(row['MOF'])
            temp_dict = {'k_H': row['k_H'], 'Maximum Composition': row['Maximum Composition']}
            all_kH[row['MOF']][gas] = temp_dict

    filename_all_kH = filepath+'all_kH.csv'
    with open(filename_all_kH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for mof in mof_list:
            writer.writerow([ '{\''+mof+'\': '+str(all_kH[mof])+'}' ])

    return all_kH


def read_all_kH_results(filename):
    import ast
    with open(filename, newline='') as csvfile:
        output_data = csv.reader(csvfile, delimiter="\t")
        output_data = list(output_data)
        for i in range(len(output_data)):
            for j in range(len(output_data[i])):
                output_data[i][j] = ast.literal_eval(output_data[i][j])
        final_dict = {}
        for row in output_data:
            final_dict={**final_dict, **row[0]}

        return final_dict


def calculate_all_arrays_list(mof_list, num_mofs):
    mof_array_list = []
    array_size = min(num_mofs)
    while array_size <= max(num_mofs):
        mof_array_list.extend(list(combinations(mof_list, array_size)))
        array_size += 1

    return mof_array_list


def create_henrys_matrix(array, gases, all_kH):
    H = []
    for mof in array:
        temp_row = []
        for gas in gases:
            temp_row.extend([all_kH[mof][gas+'_kH']])
        H.extend([temp_row])
    return H



# Load MOF List
filepath = 'config_files/process_config.sample.yaml'
data = yaml_loader(filepath)
mof_list = data['mof_list']

# # Load and/or Create All_kH dict
# filepath = '/Users/brian_day/Desktop/HC_work/HenrysConstants_Analysis_Results/'
# gases = ['acetone', 'ammonia', 'argon', 'CO2', 'hydrogen']
# filename = '_henrys_coefficients.csv'
# filename_all_kH = filepath+'all_kH.csv'
# all_kH=read_all_kH_results(filename_all_kH)

# ----- Load Henry's Coefficient Data ----
gases = ['acetone', 'ammonia', 'argon', 'CO2', 'hydrogen']
gases = ['ammonia', 'argon', 'CO2']
figure_path = '/Users/brian_day/Desktop/HC_Work/HenrysConstants_Analysis_Results/'

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
data_hg_all, data_air_all, data_combo_all = load_henrys_data(figure_path, gases)

# ----- Average the pure air mass for all MOFs, Filter MOFs by Henry's Regime -----
gases = ['ammonia', 'argon', 'CO2']
figure_path = '/Users/brian_day/Desktop/HC_Work/HenrysConstants_Analysis_Results/'
data_hg_all, data_air_all, data_combo_all = load_henrys_data(figure_path, gases)

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
mof_list_filtered, data_combo_avg = filter_mof_list_and_average_pure_air_masses(mof_list, data_combo_all, min_allowed_comp=0.05)

# # Filter out MOFs where no ammonia was simulated
# mof_list_filtered_twice = []
# for row in breath_sample_full:
#     if row['ammonia_mass'] != 0:
#         mof_list_filtered_twice.extend([row['MOF'].split('_')[0]])
# mof_list_filtered = mof_list_filtered_twice


# ----- Reformat Henry's Data for Conveneince -----
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
data_hg_reformatted, data_air_reformatted, data_combo_reformatted = reformat_henrys_data(mof_list_filtered, data_hg_all, data_air_all, data_combo_avg)




# ----- Screen Arrays -----
# For now, it is convenient to use the built in SVD, in the future may need to write it ourselves
# for more flexibility in handling it for array design specifically.
# Initially, use the same 'screening' method as where arrays are ranked based on the smallest
# singular value (largest smallest singular value = best, smallest smallest singular vlaue = worst).
def calculate_all_arrays_list(mof_list, num_mofs):
    mof_array_list = []
    array_size = min(num_mofs)
    while array_size <= max(num_mofs):
        mof_array_list.extend(list(itertools.combinations(mof_list, array_size)))
        array_size += 1

    return mof_array_list
list_of_arrays = calculate_all_arrays_list(mof_list_filtered, [10,10])
array = list_of_arrays[0]

Sig_list = []
for array in list_of_arrays:
    H = create_henrys_matrix(array, ['argon', 'ammonia', 'CO2'], data_combo_reformatted)
    U, Sig, V = np.linalg.svd(H)
    Sig_list.extend([Sig])

Sig_list_sorted_w_index = (sorted(list(enumerate(Sig_list)), key = lambda x: x[1][-1], reverse=True))
array_list_sorted_best_to_worst = []
for row in Sig_list_sorted_w_index:
    index = row[0]
    array_list_sorted_best_to_worst.extend([list_of_arrays[index]])
best_array = array_list_sorted_best_to_worst[0]
worst_array = array_list_sorted_best_to_worst[-1]
Sig_list_sorted_w_index[0]
Sig_list_sorted_w_index[-1]
Sig_list_sorted_w_index[int(0.5*len(list_of_arrays))]
Sig_list_sorted_w_index[1]
list_of_arrays[392283]
# Print Henry's Coefficent Data
H_best = create_filtered_henrys_matrix(best_array, ['acetone', 'ammonia', 'CO2'], all_kH, 1e-3)
for mof in best_array:
    print(mof+' = '+str(all_kH[mof]))
