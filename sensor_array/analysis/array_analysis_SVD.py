import itertools
import numpy as np
import pandas as pd
import yaml

def yaml_loader(filepath):
    with open(filepath, 'r') as yaml_file:
        data = yaml.load(yaml_file)
    return(data)


def load_filter_unite_henrys_data(filepath, gases, min_comp=0.0):
    # ddf = dictionary of dataframes
    ddf = {gas: pd.read_csv("%s%s.csv" %(filepath, gas), sep='\t', engine='python') for gas in gases}
    ddf = {gas: ddf[gas][(np.isnan(ddf[gas].kh) == False) & (ddf[gas].max_comp >= 0.05)] for gas in gases}
    all_mofs = [set(ddf[gas]['MOF']) for gas in gases][0]
    common_mofs = all_mofs[0].intersection(*all_mofs[1::])
    ddf = {gas: ddf[gas][ddf[gas].MOF.isin(common_mofs)] for gas in gases}

    df_unified = pd.DataFrame(common_mofs, columns=['MOF'])
    for gas in gases:
        df_unified[gas] = list(ddf[gas]['gas_kh'])
    mass_constants = [list(ddf[gas]['pure_air_mass']) for gas in gases]
    avg_mass_constants = np.sum(mass_constants, axis=0)/np.shape(mass_constants)[0]
    df_unified['pure_air_mass'] = avg_mass_constants

    return df_unified, common_mofs


def calculate_all_arrays(mof_list, array_size):
    mof_array_list = list(itertools.combinations(mof_list, array_size))

    return mof_array_list


def create_henrys_matrix(array, gases, kh_dataframe):
    df = kh_dataframe
    h_matrix = [list(df.loc[df['MOF'].isin(array)][gas]) for gas in gases]

    return h_matrix


def rank_arrays(list_of_arrays, gases, kh_dataframe):
    sig_list = []
    for array in list_of_arrays:
        h_matrix = create_henrys_matrix(array, gases, kh_dataframe)
        u_matrix, sig_matrix, v_matrix = np.linalg.svd(h_matrix)
        sig_list.extend([sig_matrix])

    sig_list_sorted_w_index = (sorted(list(enumerate(sig_list)), key = lambda x: x[1][-1], reverse=True))
    array_list_sorted_best_to_worst = []
    for row in sig_list_sorted_w_index:
        index = row[0]
        array_list_sorted_best_to_worst.extend([list_of_arrays[index]])

    return array_list_sorted_best_to_worst


def execute_array_analysis(config_file):
    data = yaml_loader(config_file)
    filepath = data['filepath']
    gases = data['gases']
    array_size = data['array_size']
    min_comp = data['minimum_composition']

    kh_df, common_mofs = load_filter_unite_henrys_data(filepath, gases, min_comp=0.05)
    list_of_arrays = calculate_all_arrays(common_mofs, array_size)
    array_list_sorted_best_to_worst = rank_arrays(list_of_arrays, gases, kh_df)
