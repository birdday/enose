import numpy as np
import csv
import yaml


def yaml_loader(filepath):
    with open(filepath, 'r') as yaml_file:
        data = yaml.load(yaml_file)
    return data


def create_henrys_comp_list(trace_gases, trace_gas_comps, background_gases, background_gas_types, background_gas_values, filepath=None):

    relative_ratios_total = np.sum([background_gas_values[i] for i in range(len(background_gases)) if background_gas_types[i] == 'relative'])

    comp_list = []
    for trace_gas_comp in trace_gas_comps:
        bg_comps_dict = {gas: 0 for gas in background_gases}
        for i in range(len(background_gases)):
            gas = background_gases[i]
            type = background_gas_types[i]
            value = background_gas_values[i]
            if type == 'fixed':
                bg_comps_dict[gas] = value

        total_comp = np.sum(list(bg_comps_dict.values()))+trace_gas_comp
        for i in range(len(background_gases)):
            gas = background_gases[i]
            type = background_gas_types[i]
            value = background_gas_values[i]
            if type == 'relative':
                bg_comps_dict[gas] = (1-total_comp)*value/relative_ratios_total
        comp_set = [trace_gas_comp] + [bg_comps_dict[gas] for gas in background_gases]
        comp_list.extend([comp_set])

    if filepath != None:
        for gas in trace_gases:
            gases = [gas]+background_gases
            filename = filepath+gas+'.csv'
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(gases)
                writer.writerows(comp_list)

    return comp_list


def execute_create_henrys_comp_list(config_file):
        data = yaml_loader(config_file)
        trace_gases = data['trace_gases']
        trace_gas_comps = data['trace_gas_comps']
        background_gases = list(data['background_gases'].keys())
        background_gas_types = [data['background_gases'][gas]['Type'] for gas in background_gases]
        background_gas_values = [data['background_gases'][gas]['Value'] for gas in background_gases]
        filepath = data['filepath']
        create_henrys_comp_list(trace_gases, trace_gas_comps, background_gases, background_gas_types, background_gas_values, filepath=filepath)
