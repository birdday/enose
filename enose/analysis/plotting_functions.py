import copy
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np


def plot_predicted_mass_for_each_MOF(adsorbed_masses, adsorbed_masses_error, predicted_masses, gases, known_comp, mof_list, filepath=None, sample_number=None):

    gases_w_total = copy.deepcopy(gases)+['Total']
    for gas in gases_w_total:
        x = np.linspace(1,len(adsorbed_masses[gas]),len(adsorbed_masses[gas]))
        if gas != 'Total':
            bs_str = 'Mole Fraction = '+str(np.round(known_comp[gas+'_comp'],7))
        else:
            bs_str = 'Mole Fraction = 1'
        plt.figure(figsize=(5,5), dpi=600)
        plt.errorbar(x,adsorbed_masses[gas],adsorbed_masses_error[gas], marker='o', markersize=3, elinewidth=1, linewidth=0, alpha=0.7,label='Simulated')

        predicted_masses_midpoint = [0.5*np.sum(val) for val in predicted_masses[gas]]
        predicted_masses_error = [abs(0.5*np.subtract(*val)) for val in predicted_masses[gas]]
        plt.errorbar(x+0.25,predicted_masses_midpoint,predicted_masses_error, marker='o', markersize=3, elinewidth=1, linewidth=0, alpha=0.7,label='Predicted')

        plt.xticks(range(1,len(mof_list)+1), mof_list, rotation=45, ha='right', fontsize=8)
        plt.ylabel('Adsorbed Mass [mg/g Framework]')
        plt.grid(alpha=0.3)
        plt.legend(loc='upper left')
        if filepath != None:
            if sample_number != None:
                plt.title('Breath Sample #'+str(sample_number)+'\n'+gas+', '+bs_str)
                plt.tight_layout()
                plt.savefig(filepath+'Sample_'+str(sample_number)+'/'+gas+'_mass.png')
            else:
                plt.title(gas+'\n'+bs_str)
                plt.tight_layout()
                plt.savefig(filepath+gas+'_mass.png')
        plt.close()


def plot_algorithm_progress_single_samples(gases, all_comp_sets, true_comp, cycle_nums, run_id, filepath):
    colors = mpl.cm.get_cmap('RdBu')
    color0 = colors(0.80)
    color1 = colors(0.20)

    for gas in gases:
        comp_range = all_comp_sets[gas]
        true_comp_as_array = true_comp[gas+'_comp']*np.ones(len(cycle_nums))
        plt.figure(figsize=(4.5,4.5), dpi=300)
        plt.rcParams['font.size'] = 12

        if gas == 'CO2':
            gas_for_title = '$Carbon$'+' '+'$Dioxide$'
        else:
            gas_for_title = '$' + gas[0].upper() + gas[1::] + '$'

        plt.title(gas_for_title, fontsize=16)
        plt.plot(cycle_nums, true_comp_as_array, '--', color='dimgrey', label='True Composition')
        plt.xlabel('Cycle Number', fontsize=16)
        xticks = [i for i in range(len(cycle_nums)) if i % 2==0]
        plt.xticks(xticks, fontsize=12)
        plt.ylabel('Mole Fraction', fontsize=16)
        plt.yticks(fontsize=12)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(-1,3), useMathText=True)
        for n in range(len(cycle_nums)):
            if n == 0:
                plt.plot([cycle_nums[n],cycle_nums[n]], comp_range[n], 'o-', color=color0, label='Predicted')
            else:
                plt.plot([cycle_nums[n],cycle_nums[n]], comp_range[n], 'o-', color=color0, label=None)
        plt.legend(fontsize=12, bbox_to_anchor=(0.5,-0.15), loc='upper center', ncol=2)
        plt.tight_layout(rect=(0,0.05,1,1))
        plt.savefig(filepath+'breath_sample_prediction_algorithm_'+gas+'.png')
        plt.close()


def plot_predicted_vs_true_for_all_breath_samples(gases, gas_limits, predicted_comps, true_comps, filepath=None, sort_data=False, sort_by=None):
    from matplotlib.ticker import AutoMinorLocator
    from matplotlib.ticker import FixedLocator

    colors = mpl.cm.get_cmap('RdBu')
    color0 = colors(0.80)
    color1 = colors(0.20)

    xlim_value = int(len(predicted_comps)+1)
    predicted_comps_copy = copy.deepcopy(predicted_comps)
    for gas in gases:
        plt.figure(figsize=(4.5,4.5), dpi=600)
        plt.rcParams['font.size'] = 12
        plt.ticklabel_format(axis="y", style="sci", scilimits=(-1,0), useMathText=True)
        plt.xlim([0,xlim_value])
        plt.xlabel('Sample Number', fontsize=16)
        plt.xticks(fontsize=12)
        plt.ylim([gas_limits[gas][0],gas_limits[gas][1]*1.00])
        plt.ylabel('Mole Fraction', fontsize=16)
        plt.yticks(fontsize=12)
        if gas == 'CO2':
            gas_for_title = '$Carbon$'+' '+'$Dioxide$'
        else:
            gas_for_title = '$' + gas[0].upper() + gas[1::] + '$'
        plt.title(gas_for_title, fontsize=16)

        x = np.linspace(1,len(predicted_comps),len(predicted_comps))
        true_comps_by_component = [row[gas+'_comp'] for row in true_comps]

        predicted_comps = predicted_comps_copy
        if sort_data == True:
            if sort_by == 'all':
                sorted_indicies = list(np.argsort(true_comps_by_component))
                true_comps_by_component_temp = list(sorted(true_comps_by_component))
                true_comps_by_component = true_comps_by_component_temp
                predicted_comps_temp = [predicted_comps[index] for index in sorted_indicies]
                predicted_comps = predicted_comps_temp
            elif sort_by != None:
                true_comps_sort_by = [row[sort_by+'_comp'] for row in true_comps]
                sorted_indicies = list(np.argsort(true_comps_sort_by))
                true_comps_by_component_temp = [true_comps_by_component[index] for index in sorted_indicies]
                true_comps_by_component = true_comps_by_component_temp
                predicted_comps_temp = [predicted_comps[index] for index in sorted_indicies]
                predicted_comps = predicted_comps_temp
            else:
                raise(NameError('Invalid Gas for sorting!'))

        plt.plot(x,true_comps_by_component, marker='o', markersize=4, linewidth=0, alpha=0.7, color=color0, label='True')

        predicted_comps_midpoint = [0.5*np.sum(row[gas]) for row in predicted_comps]
        predicted_comps_error = [abs(0.5*np.subtract(*row[gas])) for row in predicted_comps]
        plt.errorbar(x,predicted_comps_midpoint,predicted_comps_error, marker='o', markersize=4, elinewidth=1, linewidth=0, alpha=0.7, color=color1, label='Predicted')

        minor_locator1 = FixedLocator(np.linspace(5,xlim_value-1,10))
        plt.gca().xaxis.set_minor_locator(minor_locator1)
        plt.grid(which='minor', alpha=0.3)
        plt.legend(fontsize=12, bbox_to_anchor=(0.5,-0.15), loc='upper center', ncol=2, markerscale=1.5)
        plt.tight_layout(rect=(0,0.05,1,1))
        if filepath != None:
            filename = 'breath_sample_prediciton_'+gas+'.png'
            plt.savefig(filepath+filename)
        plt.close()


def plot_prediction_error_for_all_breath_samples(gases, predicted_comps, true_comps, filepath=None):
    from matplotlib.ticker import AutoMinorLocator
    from matplotlib.ticker import FixedLocator

    colors = mpl.cm.get_cmap('RdBu')
    color0 = colors(0.80)
    color1 = colors(0.20)

    xlim_value = int(len(predicted_comps)+1)
    for gas in gases:
        plt.figure(figsize=(5,5), dpi=600)
        plt.xlim([0,xlim_value])
        plt.xlabel('Sample Number', fontsize=16)
        plt.xticks(fontsize=12)
        # plt.ylim(gas_limits[gas])
        plt.ylabel('Percent Error', fontsize=16)
        plt.yticks(fontsize=12)
        if gas == 'CO2':
            gas_for_title = '$Carbon$'+' '+'$Dioxide$'
        else:
            gas_for_title = '$' + gas[0].upper() + gas[1::] + '$'
        plt.title(gas_for_title, fontsize=16)

        x = np.linspace(1,len(predicted_comps),len(predicted_comps))
        true_comps_by_component = [row[gas+'_comp'] for row in true_comps]
        true_value_error = np.zeros(len(true_comps_by_component))
        plt.plot(x,true_value_error, marker='o', markersize=4, linewidth=0, alpha=0.7, color=color0, label='True')

        predicted_comps_midpoint = [0.5*np.sum(row[gas]) for row in predicted_comps]
        predicted_comps_error = [abs(0.5*np.subtract(*row[gas])) for row in predicted_comps]
        percent_error_midpoint = np.divide(np.add(predicted_comps_midpoint,np.multiply(-1,true_comps_by_component)),true_comps_by_component)*100
        percent_error_error = np.divide(predicted_comps_error,true_comps_by_component)*100
        if max(abs(percent_error_midpoint))+max(percent_error_error) <= 5:
            plt.ylim([-5,5])
        plt.errorbar(x,percent_error_midpoint,percent_error_error, marker='o', markersize=4, elinewidth=1, linewidth=0, alpha=0.7, color=color1, label='Predicted')

        minor_locator1 = FixedLocator(np.linspace(5,xlim_value-1,10))
        plt.gca().xaxis.set_minor_locator(minor_locator1)
        plt.grid(which='minor', alpha=0.3)
        plt.legend(bbox_to_anchor=(0.5,-0.15), loc='upper center', ncol=2, fontsize=12)
        plt.tight_layout(rect=(0,0.05,1,1))
        if filepath != None:
            filename = 'breath_sample_prediction_error_'+gas+'.png'
            plt.savefig(filepath+filename)
        plt.close()


def plot_kld_progression_w_max_pmf(all_array_pmfs_nnempf, all_array_pmfs_normalized, cycle_nums, figname=None):
    colors = mpl.cm.get_cmap('RdBu')
    color0 = colors(0.80)
    color1 = colors(0.20)

    # Calculate KLD Values for each cycle
    kld_values = []
    # for i in range(len(all_array_pmfs_normalized)):
        # Commenting this out to try and get an initial passing build. This plotting function currently does not work since the function in the line below is not defined or imported. May be located on a non tracked copy of the file, or this was pushed only becuase of computer switch.
        # kld = calculate_KLD_for_cycle(all_array_pmfs_normalized[i])
        # kld_values.extend([kld])

    # Set up x/y data
    x_data = cycle_nums[1::]
    y_data = kld_values

    # Prepare Plot
    plt.figure(figsize=(5.5,4.5), dpi=300)
    plt.rcParams['font.size'] = 12
    plt.title('KLD and Probability', fontsize=16)
    plt.xlabel('Cycle Number', fontsize=16)
    xticks = [i for i in range(len(cycle_nums)) if i % 2==0]
    plt.xticks(xticks, fontsize=12)

    # KLD Data / Axis
    plt.ylabel('Normalized KLD', fontsize=16, color=color0)
    plt.yticks(fontsize=12, color=color0)
    plt.ylim([1e-18,2])
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(-1,3), useMathText=True)
    plt.semilogy(x_data, y_data, '-o', color=color0, label='Cycle KLD')

    # PMF Data / Axis
    plt.twinx()

    y_data = [row[0] for row in all_array_pmfs_nnempf]
    plt.semilogy(x_data, y_data, '-o', color=color1, label='Cycle Max. PMF Value')

    plt.ylabel('Max. Semi-Normalized\nPMF Value', fontsize=16, color=color1)
    plt.yticks([0.999, 1.0], fontsize=12, color=color1, rotation=90, va='center')
    plt.ylim([0.998,1.0002])
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(-1,3), useMathText=True)

    # plt.legend(fontsize=12, bbox_to_anchor=(0.5,-0.15), loc='upper center', ncol=2)
    plt.tight_layout()
    plt.savefig(figname+'KLD_vs_cycle.png')
    plt.close()


def plot_all_array_pmf(all_array_pmfs_nnempf, figname=None):
    # colors = mpl.cm.get_cmap('viridis')
    # colors_vect = colors(np.linspace(0,1,len(all_array_pmfs_nnempf)))
    colors = mpl.cm.get_cmap('RdBu')
    color0 = colors(0.80)
    color1 = colors(0.20)

    plt.figure(figsize=(4.5,4.5), dpi=300)
    plt.rcParams['font.size'] = 12
    plt.title('Semi-Normalized Array Probability', fontsize=16)
    plt.xlabel('Array PMF\nfrom Largest to Smallest', fontsize=16)
    plt.xticks([])

    for i in range(len(all_array_pmfs_nnempf)):
        y_data = all_array_pmfs_nnempf[i]
        x_data = np.linspace(0,100,len(y_data))
        plt.semilogy(x_data, y_data, label='Cycle '+str(i+1), color=color0)
    plt.ylim([1e-2,5])

    plt.ylabel('Semi-Normalized PMF', fontsize=16)
    plt.yticks(fontsize=12)
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(-1,3), useMathText=True)

    plt.legend(fontsize=12, bbox_to_anchor=(0.5,-0.15), loc='upper center', ncol=4)
    plt.tight_layout() #rect=(0,0.05,1,1)
    plt.savefig(figname+'ArrayPMF_vs_Cycle.png')
    plt.close()
