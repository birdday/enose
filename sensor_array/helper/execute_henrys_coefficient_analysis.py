# ----- Perform Analysis -----
gases = ['acetone', 'ammonia', 'argon', 'CO2', 'hydrogen']
# gases = ['CO2']
air_ratios = ['3to1air', '4to1air', '5to1air']
for gas in gases:
    # Single Ratios Test
    # for air_ratio in air_ratios:
    #     folder = gas+'_'+air_ratio+'/'
    #     sim_results_path = '/Users/brian_day/Desktop/HC_Work/HenrysConstants_Final/'+folder
    #     analysis_results_path = '/Users/brian_day/Desktop/HC_Work/HenrysConstants_Analysis_Results/'+folder
    #     if os.path.isdir(analysis_results_path) != True:
    #         os.mkdir(analysis_results_path)
    #     analyze_single_ratio(gas, air_ratio, sim_results_path, analysis_results_path)

    # All Ratio Test
    sim_results_path = '/Users/brian_day/Desktop/HC_Work/HenrysConstants_Final/'
    analysis_results_path = '/Users/brian_day/Desktop/HC_Work/HenrysConstants_Analysis_Results/'+str(gas)+'_AllRatios/'
    if os.path.isdir(analysis_results_path) != True:
        os.mkdir(analysis_results_path)
    analyze_all_ratios(gas, sim_results_path, analysis_results_path, hg_eval_type='R2', air_eval_type='With hg', r2_min_hg=0.95, r2_min_air=0.0, rmse_min_hg=0.10, rmse_min_air=0.10)

# ----- Read / Replot Henry's Coefficients -----
gases = ['acetone', 'ammonia', 'argon', 'CO2', 'hydrogen']
figure_path = '/Users/brian_day/Desktop/HC_Work/HenrysConstants_Analysis_Results/'
for gas in gases:
    filename_hg = '/Users/brian_day/Desktop/HC_Work/HenrysConstants_Analysis_Results/'+str(gas)+'_AllRatios/_henrys_coefficients_hg.csv'
    filename_air = '/Users/brian_day/Desktop/HC_Work/HenrysConstants_Analysis_Results/'+str(gas)+'_AllRatios/_henrys_coefficients_air.csv'
    data_hg = read_kH_results(filename_hg)
    data_air = read_kH_results(filename_air)
    plot_all_kH(gas, data_hg, figure_path+'All_kH_for_'+str(gas)+'.png')
    plot_all_kH(gas, data_air, figure_path+'All_kH_for_air_with_'+str(gas)+'.png')
    combo_kH = []
    for row_hg in data_hg:
        for row_air in data_air:
            if row_hg['MOF'] == row_air['MOF']:
                if row_hg['k_H'] != None and row_air['k_H'] != None:
                    combo_kH_temp = {}
                    kH_temp = row_hg['k_H'] + row_air['k_H']
                    combo_kH_temp['MOF'] = row_hg['MOF']
                    combo_kH_temp['Maximum Composition'] = row_hg['Maximum Composition']
                    combo_kH_temp['k_H'] = kH_temp
                    combo_kH.extend([combo_kH_temp])
    plot_all_kH(gas, combo_kH, figure_path+'All_kH_for_combo_air_'+str(gas)+'.png')


breath_filepath = '/Users/brian_day/Desktop/HC_Work/breath-sample-results/healthy-samples/'
files = list(glob.glob(breath_filepath+'*/*.csv'))
_, b_test = import_simulated_data(files[0])

# Might need to reincorporate as Nan and Inf start to reappear (only needed for very bad fits)
# with open(filename, newline='') as csvfile:
#     output_data = csv.reader(csvfile, delimiter="\t")
#     output_data = list(output_data)
# full_array = []
# for i in range(len(output_data)):
#     print(i)
#     row = output_data[32][0]
#     row = row.replace('nan', '\'nan\'')
#     row = row.replace('inf', '\'inf\'')
#     row = row.replace('-\'inf\'', '\'-inf\'')
#     temp_array = []
#     temp_row = ast.literal_eval(row)
#     temp_array.append(temp_row)
#     full_array.extend(temp_array)
