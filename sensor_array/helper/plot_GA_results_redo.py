import numpy as np
from GeneticAlgorithm_BrianEdits import *

num_runs = 4
num_generations = [25, 25, 50, 50, 50]
num_generations_total = np.sum(num_generations)
array_size_vect = [40, 45]
# array_size_vect = [1,2,3,4,5]
results_filepath = '/Users/brian_day/Desktop/GeneticAlgorithm_Results/elitist_GA/1bar/'

best_data_filepath = '/Users/brian_day/Desktop/GeneticAlgorithm_Results/elitist_GA/1Bar/best_sizes_10_to_45/'
worst_data_filepath = '/Users/brian_day/Desktop/GeneticAlgorithm_Results/elitist_GA/1Bar/worst_sizes_10_to_45/'

# For plotting all GA results simultaneously to view the approximate spread of KLD results
for i in range(len(array_size_vect)):
    array_size = array_size_vect[i]

    best_kld_each_gen = []
    worst_kld_each_gen = []
    all_others_from_best_each_gen = []
    all_others_from_worst_each_gen = []
    generations = []
    generations_for_others = []

    for j in range(num_runs):
        best_data_filename = 'arraysize_%s_testnum_%s_best_by_Absolute_KLD.csv' % (array_size, j+1)
        best_data_fullname = best_data_filepath + best_data_filename
        best_data = read_GA_results_clean(best_data_fullname)

        worst_data_filename = 'arraysize_%s_testnum_%s_worst_by_Absolute_KLD.csv' % (array_size, j+1)
        worst_data_fullname = worst_data_filepath + worst_data_filename
        worst_data = read_GA_results_clean(worst_data_fullname)

        best_kld_this_gen = []
        worst_kld_this_gen = []
        all_others_from_best_this_gen = []
        all_others_from_worst_this_gen = []
        generations_this_gen = []
        generations_for_others_this_gen = []

        for k in range(num_generations_total):
            worst_kld_temp = worst_data[k][0]['Absolute_KLD']
            worst_kld_this_gen.append(worst_kld_temp)

            best_kld_temp = best_data[k][0]['Absolute_KLD']
            best_kld_this_gen.append(best_kld_temp)

            for m in best_data[k][1:]:
                all_others_from_best_temp = m['Absolute_KLD']
                all_others_from_best_this_gen.append(all_others_from_best_temp)
                generations_for_other_temp = m['Generation']
                generations_for_others_this_gen.append(generations_for_other_temp)

            for m in worst_data[k][1:]:
                all_others_from_worst_temp = m['Absolute_KLD']
                all_others_from_worst_this_gen.append(all_others_from_worst_temp)

            current_gen = best_data[k][0]['Generation']
            generations_this_gen.append(current_gen)

        best_kld_each_gen.extend(best_kld_this_gen)
        worst_kld_each_gen.extend(worst_kld_this_gen)
        all_others_from_best_each_gen.extend(all_others_from_best_this_gen)
        all_others_from_worst_each_gen.extend(all_others_from_worst_this_gen)
        generations.extend(generations_this_gen)
        generations_for_others.extend(generations_for_others_this_gen)

    fig_filename = 'arraysize_%s.png' % (array_size)
    fig_fullname = results_filepath + fig_filename

    plt.clf()
    plt.figure(figsize=[6,4], dpi=500)

    plt.plot(generations_for_others, all_others_from_best_each_gen, 'o', color='#dddddd', alpha=0.03)
    plt.plot(generations_for_others, all_others_from_worst_each_gen, 'o', color='#dddddd', alpha=0.03)
    plt.plot(generations, best_kld_each_gen, 'o', color='#1f77b4', alpha=1.0)
    plt.plot(generations, worst_kld_each_gen, 'o', color='#ff7f0e', alpha=1.0)
    plt.legend(['All Others', '_nolegend_','Best KLD','Worst KLD'], loc='upper left')

    for i in range(1,len(num_generations)):
        plt.axvline(x=np.sum(num_generations[0:i])+0.5, c='black', ls='dashed', lw=1)

    plt.xlabel('Generation', Fontsize=16)
    plt.xlim(0,num_generations_total)
    plt.xticks(np.linspace(0,num_generations_total,11))

    plt.ylabel('Absolute KLD', Fontsize=16)
    plt.ylim(0,10)
    plt.yticks(np.linspace(0,10,11))

    plt.title('Genetic Algorithm Approach \n Array Size = %s' % array_size, Fontsize=16)
    plt.tight_layout()
    plt.savefig(fig_fullname)


# # For plotting the best/worst arrays vs their MOF properties
#
# # Import yaml file as dictoncary
# filepath = '/Users/brian_day/Github-Repos/Sensor_Array/settings/process_config.extra_properties.yaml'
# data = yaml_loader(filepath)
#
# # Redefine key varaibles in yaml file
# num_mofs = data['number_mofs']
# num_mixtures = data['num_mixtures']
# num_bins = data['num_bins']
# num_best_worst = data['num_best_worst']
# stdev = data['stdev']
# mrange = data['mrange']
# gases = data['gases']
# mof_list = data['mof_list']
#
# # Physical Properties
# mof_density = {}
# mof_lcd = {}    # largest_cavity_diameter
# mof_pld = {}    # pore_limiting_diameter
# mof_vsa = {}    # volumetric_surface_area
# mof_gsa = {}    # gravimetric_surface_area
# mof_vf = {}     # void_fraction
#
# for mof in mof_list:
#     mof_density[mof] = data['mofs'][mof]['density']
#     mof_lcd[mof] = data['mofs'][mof]['largest_cavity_diameter']
#     mof_pld[mof] = data['mofs'][mof]['pore_limiting_diameter']
#     mof_vsa[mof] = data['mofs'][mof]['volumetric_surface_area']
#     mof_gsa[mof] = data['mofs'][mof]['gravimetric_surface_area']
#     mof_vf[mof] = data['mofs'][mof]['void_fraction']
#
# def get_best_and_worst_arrays(array_size, num_runs, num_arrays):
#     all_arrays_all_runs_best = []
#     all_arrays_all_runs_worst = []
#
#     for j in range(num_runs):
#         best_data_filename = 'arraysize_%s_testnum_%s_best.csv' % (array_size, j+1)
#         best_data_fullname = best_data_filepath + best_data_filename
#         best_data = read_GA_results_clean(best_data_fullname)
#         worst_data_filename = 'arraysize_%s_testnum_%s_worst.csv' % (array_size, j+1)
#         worst_data_fullname = worst_data_filepath + worst_data_filename
#         worst_data = read_GA_results_clean(worst_data_fullname)
#
#         for k in range(2*num_generations):
#             all_arrays_this_gen_best = best_data[k]
#             all_arrays_all_runs_best.extend(all_arrays_this_gen_best)
#             all_arrays_this_gen_worst = worst_data[k]
#             all_arrays_all_runs_worst.extend(all_arrays_this_gen_worst)
#
#     all_arrays_all_runs_ordered_by_best = sorted(all_arrays_all_runs_best, key=lambda k: k['Absolute_KLD'], reverse=True)
#     all_arrays_all_runs_ordered_by_worst = sorted(all_arrays_all_runs_worst, key=lambda k: k['Absolute_KLD'], reverse=False)
#
#     all_arrays_all_runs_ordered_by_best_filtered = [all_arrays_all_runs_ordered_by_best[0]]
#     all_arrays_all_runs_ordered_by_worst_filtered = [all_arrays_all_runs_ordered_by_worst[0]]
#     for j in range(1,len(all_arrays_all_runs_ordered_by_best)):
#         if all_arrays_all_runs_ordered_by_best[j]['MOF_Array'] != all_arrays_all_runs_ordered_by_best[j-1]['MOF_Array']:
#             all_arrays_all_runs_ordered_by_best_filtered.extend([all_arrays_all_runs_ordered_by_best[j]])
#         if all_arrays_all_runs_ordered_by_worst[j]['MOF_Array'] != all_arrays_all_runs_ordered_by_worst[j-1]['MOF_Array']:
#             all_arrays_all_runs_ordered_by_worst_filtered.extend([all_arrays_all_runs_ordered_by_worst[j]])
#
#     best_arrays = all_arrays_all_runs_ordered_by_best_filtered[0:num_arrays]
#     worst_arrays = all_arrays_all_runs_ordered_by_worst_filtered[0:num_arrays]
#
#     return(best_arrays, worst_arrays)
#
#
# def plot_properties_of_arrays(best_arrays, worst_arrays, mof_property, mof_property_for_fig_name, xlim=None, xticks=None, ylim=None, yticks=None):
#     xvals_best = []
#     yvals_best =[]
#     for j in range(len(best_arrays)):
#         for mof in best_arrays[j]['MOF_Array']:
#             xvals_best.extend([j+1])
#             yvals_best.extend([mof_property[mof]])
#
#     xvals_worst = []
#     yvals_worst =[]
#     for j in range(len(worst_arrays)):
#         for mof in worst_arrays[j]['MOF_Array']:
#             xvals_worst.extend([j+1])
#             yvals_worst.extend([mof_property[mof]])
#
#     fig_filename = 'arraysize_%s_%s_plot.png' % (array_size, mof_property_for_fig_name)
#     fig_fullname = results_filepath + fig_filename
#
#     plt.clf()
#     fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
#     fig.set_size_inches([6,4])
#     fig.set_dpi(600)
#     fig.suptitle('Physical Property Analysis: %s\nArray Size: %s' % (mof_property_for_fig_name, array_size)).set_size(14)
#     fig.text(0.5, 0.02, 'Ordered Arrays', ha='center', Fontsize=12)
#     fig.text(0.02, 0.5, 'MOF %s' % (mof_property_for_fig_name), va='center', rotation='vertical', Fontsize=12)
#
#     ax[0].plot(xvals_best, yvals_best, 'o', color='grey', alpha=1.0, markersize=2)
#     ax[0].set_title('Best Arrays', Fontsize=12)
#     if xlim != None:
#         ax[0].set_xlim(xlim)
#         ax[0].xaxis.set_ticks(xticks)
#     if ylim != None:
#         ax[0].set_ylim(ylim)
#         ax[0].yaxis.set_ticks(yticks)
#
#     ax[1].plot(xvals_worst, yvals_worst, 'o', color='grey', alpha=1.0, markersize=2)
#     ax[1].set_title('Worst Arrays', Fontsize=12)
#     if xlim != None:
#         ax[1].set_xlim(xlim)
#         ax[1].xaxis.set_ticks(xticks)
#     if ylim != None:
#         ax[1].set_ylim(ylim)
#         ax[1].yaxis.set_ticks(yticks)
#
#     plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.9])
#     plt.savefig(fig_fullname)
#
# num_runs=5
# num_arrays=50
# for i in range(len(array_size_vect)):
#     array_size = array_size_vect[i]
#     best, worst = get_best_and_worst_arrays(array_size, num_runs, num_arrays)
#     plot_properties_of_arrays(best, worst, mof_vf, 'VF', xlim=[0,50], xticks=[0,10,20,30,40,50]) #, ylim=[0,4], yticks=[0,1,2,3,4])
