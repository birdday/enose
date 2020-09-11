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

gases = ['acetone', 'ammonia', 'argon', 'CO2', 'hydrogen']

