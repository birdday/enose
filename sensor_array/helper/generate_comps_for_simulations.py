import numpy as np
import csv
import yaml


def yaml_loader(filepath):
    with open(filepath, 'r') as yaml_file:
        data = yaml.load(yaml_file)
    return data


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


def execute_create_henrys_comp_list(config_file):
        data = yaml_loader(config_file)
        trace_gases = data['trace_gases']
        trace_gas_comps = data['trace_gas_comps']
        background_gases = list(data['background_gases'].keys())
        background_gas_ratios = [data['background_gases'][gas]['Value'] for gas in background_gases]
        filename = data['filename']
        for gas in background_gases:
            create_henrys_comp_list([gas], trace_gas_comps, background_gases, background_gas_ratios, filename=filename)
