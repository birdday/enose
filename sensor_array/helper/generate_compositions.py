import numpy as np
import itertools
import csv
import random
import scipy.stats as stats


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


def create_henrys_comp_list(henrys_gases, henrys_comps, background_gases, background_gas_ratios, filename=None):
    bg_gas_total = np.sum(background_gas_ratios)
    comp_list = []
    for i in range(len(henrys_comps)):
        hg_comp = henrys_comps[i]
        bg_comp = [(1-hg_comp)*ratio/bg_gas_total for ratio in background_gas_ratios]
        comp_list_temp = [hg_comp]
        comp_list_temp.extend(bg_comp)
        comp_list.append(comp_list_temp)

    if filename != None:
        gases = henrys_gases+background_gases
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(gases)
            writer.writerows(comp_list)

    return comp_list


def convert_dict_to_molefrac(gases_dict):
    for key in gases_dict:
        if 'Units' in gases_dict[key]:
            if gases_dict[key]['Units'] == 'percent':
                gases_dict[key]['Values'] = [val*0.01 for val in gases_dict[key]['Values']]
                gases_dict[key]['Units'] = 'molefrac'
            if gases_dict[key]['Units'] == 'ppm':
                gases_dict[key]['Values'] = [val*1e-6 for val in gases_dict[key]['Values']]
                gases_dict[key]['Units'] = 'molefrac'
            if gases_dict[key]['Units'] == 'ppb':
                gases_dict[key]['Values'] = [val*1e-9 for val in gases_dict[key]['Values']]
                gases_dict[key]['Units'] = 'molefrac'

    return gases_dict


def create_breath_samples(gases_dict, num_samples, filename=None, randomseed=69):
    """
    Dictionary format for the gases accepts the following keys:
        Gas - Name of Gas (should match RASPA name or alias)
        Type - Specifies how the gas should be handled when creating compositions
            bound - gas composition will follow a random unifrom sampling between the provided range
            stddev - gas composition will follow a normal distirbution centered around the first value with a standard deviation of the second value
            background - gas composition will be determined at the end in order to satisfy a total mole fraction of 1;
              currently only supports a single background gas, the rest should be specified by background-r
            ratio - gas composoition will be fall within a fixed range realtive to the composition of another gas
            background-r - gas compositon will be used to satsify total mole fraction equal to 1 and be within a fixed range relative to another gas
        Values -  Values will be interpreted differently depending on the specified type
            bound, background - Upper/lower bound on possible gas compositions
            stddev - first value is mean, second value is deviation
            ratio, background-r - Upper/lower bound on ratio of gas compositon, given as x/1
        Units - Specifies units used to interpret the input values, all output files will be written as mole_fraction;
          Currently accepted units are: ppb, ppm, percent, molefrac; This keyword is not necessary for ratio and background-r
        Gas-r - Used only for ratio and background-r, this keyword specfies which gas this should be determined relative to

    Gases will be processed in the following order: bounds = stddev > ratio > background > background-r
    """

    # Extract gases list from dictionary keys
    all_gases = list(gases_dict.keys())
    random.seed(randomseed)

    # Separate gases by type for handling
    gases = []
    ratio_gases = []
    background_gases = []
    background_r_gases = []
    for gas in all_gases:
        row = gases_dict[gas]
        if row['Type'] == 'bounds' or row['Type'] == 'stddev':
            gases.extend([gas])
        elif row['Type'] == 'ratio':
            ratio_gases.extend([gas])
        elif row['Type'] == 'background':
            background_gases.extend([gas])
        elif row['Type'] == 'background-r':
            background_r_gases.extend([gas])
        else:
            raise NameError('Invalid type for '+gas+'!')

    # Create composition list
    comp_list = []
    for i in range(num_samples):

        # Generate compositions for non-background gases
        temp_dict = {}
        for gas in gases:
            row = gases_dict[gas]
            v1, v2 = row['Values']
            if row['Type'] == 'bounds':
                temp_dict[gas] = random.uniform(v1, v2)
            elif row['Type'] == 'stddev':
                # temp_dict[gas] = random.gauss(v1,v2)
                lower_bound = 0
                upper_bound = 1
                mu = v1
                sigma = v2
                a = (lower_bound-mu)/sigma
                b = (upper_bound-mu)/sigma
                temp_dict[gas] = stats.truncnorm.rvs(a,b,loc=mu, scale=sigma, size=1)[0]

        for gas in ratio_gases:
            row = gases_dict[gas]
            rcomp = temp_dict[row['Gas-r']]
            temp_dict[gas] = random.uniform(rcomp*v1, rcomp*v2)

        # Generate (relative) compositions for background gases
        temp_dict_2 = {}
        for gas in background_gases:
            row = gases_dict[gas]
            temp_dict_2[gas] = 1

        for gas in background_r_gases:
            row = gases_dict[gas]
            rcomp = temp_dict_2[row['Gas-r']]
            v1, v2 = row['Values']
            ratio = random.uniform(v1, v2)
            comp = rcomp*ratio
            temp_dict_2[gas] = comp

        # Normalize background gas compositions and add to temp_dict
        total = sum(temp_dict.values())
        total_2 = sum(temp_dict_2.values())
        for key in temp_dict_2:
            temp_dict_2[key] = (temp_dict_2[key]/total_2)*(1-total)
            temp_dict[key] = temp_dict_2[key]

        comp_list.extend([temp_dict])

    # Write to file is desired
    if filename != None:

        gases = list(comp_list[0].keys())
        comp_list_only_values = []
        for row in comp_list:
            temp_array = []
            temp_array.extend([row[gas] for gas in gases])
            comp_list_only_values.extend([temp_array])

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(gases)
            writer.writerows(comp_list_only_values)

    return comp_list


# ===== Examples / Working Section =====

# ----- Create Liver Disease Test Sets -----
healthy_dict = {
                'N2':      {'Type': 'background-r', 'Values': [3, 5], 'Gas-r': 'O2'},
                'O2':       {'Type': 'background'},
                'CO2':      {'Type': 'bounds', 'Values': [2,5], 'Units': 'percent'},
                'Ammonia':  {'Type': 'stddev', 'Values': [0.49,0.08], 'Units': 'ppm'},
                'Argon':    {'Type': 'bounds', 'Values': [0.6,1.2], 'Units': 'percent'}
                }

healthy_dict_alt = {
                'N2':      {'Type': 'background-r', 'Values': [3, 5], 'Gas-r': 'O2'},
                'O2':       {'Type': 'background'},
                'CO2':      {'Type': 'stddev', 'Values': [4.15,0.82], 'Units': 'percent'},
                'Ammonia':  {'Type': 'stddev', 'Values': [0.49,0.08], 'Units': 'ppm'},
                'Argon':    {'Type': 'stddev', 'Values': [0.9,0.1], 'Units': 'percent'}
                }

diseased_dict = {
                'N2':      {'Type': 'background-r', 'Values': [3, 5], 'Gas-r': 'O2'},
                'O2':       {'Type': 'background'},
                'CO2':      {'Type': 'bounds', 'Values': [2,5], 'Units': 'percent'},
                'Ammonia':  {'Type': 'stddev', 'Values': [3.32,2.19], 'Units': 'ppm'},
                'Argon':    {'Type': 'bounds', 'Values': [0.6,1.2], 'Units': 'percent'}
                }

diseased_dict_alt = {
                'N2':      {'Type': 'background-r', 'Values': [3, 5], 'Gas-r': 'O2'},
                'O2':       {'Type': 'background'},
                'CO2':      {'Type': 'stddev', 'Values': [4.15,0.82], 'Units': 'percent'},
                'Ammonia':  {'Type': 'stddev', 'Values': [3.32,2.19], 'Units': 'ppm'},
                'Argon':    {'Type': 'stddev', 'Values': [0.9,0.1], 'Units': 'percent'}
                }


healthy_dict_mf = convert_dict_to_molefrac(healthy_dict_alt)
_ = create_breath_samples(healthy_dict_mf, 50, filename='/Users/brian_day/Desktop/breath_samples_healthy_alt.csv', randomseed=20)
diseased_dict_mf = convert_dict_to_molefrac(diseased_dict_alt)
_ = create_breath_samples(diseased_dict_mf, 50, filename='/Users/brian_day/Desktop/breath_samples_diseased_alt.csv', randomseed=50)


def load_breath_samples_alt(filename):
    with open(filename) as file:
        reader = csv.DictReader(file, delimiter='\t')
        reader_list = list(reader)
        for i in range(len(reader_list)):
            reader_list[i]['Run ID'] = int(i+1)
        keys = reader.fieldnames

    return keys, reader_list
# 
# filename ='/Users/brian_day/Desktop/breath_samples_diseased_alt.csv'
# _, samples = load_breath_samples_alt(filename)

# counter = 0
# for sample in samples:
#     non_air_comp_total = 0
#     for gas in ['Argon', 'Ammonia', 'CO2']:
#         non_air_comp_total += float(sample[gas])
#     if non_air_comp_total >= 0.05:
#         counter += 1
#         print(non_air_comp_total)

# # ----- Uniform Spacing List -----
# gases = ['argon', 'ammonia', 'CO2', 'O2', 'N2']
# gas_limits = [[0,0.012], [0, 1e-5], [0,0.05], [0.0,0.3]]
# spacing = [0.002, 5e-7, 0.005, 0.02]
#
# _, test = create_uniform_comp_list(gases, gas_limits, spacing, filename=None, imply_final_gas_range=True, imply_final_gas_spacing=True)
#
# gases = ['argon', 'ammonia', 'CO2', 'Air']
# gas_limits = [[0,0.012], [0, 1e-5], [0,0.05]]
# spacing = [0.002, 5e-7, 0.005]
#
# _, test = create_uniform_comp_list(gases, gas_limits, spacing, filename=None, imply_final_gas_range=False, imply_final_gas_spacing=False, filter_for_1=True)
# len(test)
# # ----- Henry's Compositions List -----
# henrys_gases = ['acetone']
# henrys_comps = np.linspace(0,0.05,11)
# # henrys_comps = np.divide(henrys_comps, 100)
# background_gases = ['CO2','O2','N2']
# background_gas_ratios = [3, 19.4, 77.6] #3% @ 4-1
# background_gas_ratios = [4, 19.2, 76.8] #4% @ 4-1
# background_gas_ratios = [5, 19.0, 76.0] #4% @ 4-1
#
# create_henrys_comp_list(henrys_gases, henrys_comps, background_gases, background_gas_ratios, filename='henrys_comps_co2_in_air_4at5to1.csv')
#
#
# breath_filepath = '/Users/brian_day/Desktop/HC_Work/breath-sample-results/'+type+'-samples/'
# all_breath_samples, all_breath_samples_joined = load_breath_samples(breath_filepath, mof_list_filtered)
# breath_filepath = '/Users/brian_day/Desktop/HC_Work/breath-sample-results/'+type+'-samples/'
# all_breath_samples, all_breath_samples_joined = load_breath_samples(breath_filepath, mof_list_filtered)
