import numpy as np
import csv

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
