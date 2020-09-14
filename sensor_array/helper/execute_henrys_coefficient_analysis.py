import glob
import henrys_coefficient_analysis as hca
import pandas as pd

gases = ['acetone', 'ammonia', 'argon', 'CO2', 'hydrogen']
columns = ['MOF', 'gas_kh', 'gas_r2', 'gas_rmse', 'air_kh', 'air_r2', 'air_rmse', 'kh', 'pure_air_mass', 'max_comp']
df = pd.DataFrame(columns=columns)

for gas in gases:
    df = pd.DataFrame(columns=columns)
    all_files = list(glob.glob('/Users/brian_day/Desktop/HC_Work/AdsorptionData_Clean/'+gas+'/*.csv'))
    for file in all_files:
        ads_data = hca.import_simulated_data(file)
        mof = ads_data['MOF'][0]
        p, max_comp, r2, rmse, i = hca.calculate_kH(ads_data, gas, eval_type='R2', r2_min=0.99, weight_type='error', fixed_intercept=True)
        if i != None:
            p_air, r2_air, rmse_air = hca.calculate_kH_air(ads_data, gas, i, weight_type='error')
            row = [mof, float(p[0]), r2, rmse, float(p_air[0]), r2_air, rmse_air, float(p[0]+p_air[0]), float(p_air[1]), max_comp]
        else:
            row = [mof, p[0], r2, rmse, None, None, None, None, None, max_comp]
        df = df.append(dict(zip(df.columns, row)), ignore_index=True)
    df.to_csv('/Users/brian_day/Desktop/HC_Work/AdsorptionData_Clean/'+gas+'.csv', sep='\t', index=False)
