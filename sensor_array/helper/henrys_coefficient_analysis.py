import glob
import numpy as np
import yaml
import pandas as pd


def yaml_loader(filepath):
    with open(filepath, 'r') as yaml_file:
        data = yaml.load(yaml_file)
    return data


def import_simulated_data(filename):
    ads_data = pd.read_csv(filename, sep='\t', engine='python')
    return ads_data


def create_weight_matrix(error, weight_type='error'):
    if weight_type == None:
        weights=np.ones(len(error))

    elif weight_type == 'error':
        # N.B. Connot simply invert value as can get a 0 error when adsorption is 0.
        # Same is true for squared error.
        weights_temp = [1/val for val in error if val != 0]
        if len(weights_temp) != 0:
            weights = np.array([1/val if val != 0 else max(weights_temp) for val in error])
        else:
            weights = np.ones(len(error))

    elif weight_type == 'error_squared':
        error_squared = [val**2 for val in error]
        weights_temp = [1/val for val in error_squared if val != 0]
        if len(weights_temp) != 0:
            weights = np.array([1/val if val != 0 else max(weights_temp) for val in error_squared])
        else:
            weights = np.ones(len(error_sqaured))

    else:
        raise NameError('Invalid Error Type!')

    return weights


def calculate_r2_and_rmse(p, x_data, y_data):
    fit = np.poly1d(p)
    predicted_vals = fit(x_data)
    residuals = predicted_vals-y_data
    ybar = np.mean(y_data)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data-ybar)**2)
    r2 = 1-(ss_res/ss_tot)
    mse = ss_res/len(y_data)
    rmse = mse**(0.5)

    return r2, rmse, ybar


def check_r2_and_rmse(r2, r2_min, rmse, rmse_min, y_bar, eval_type=None):
    if eval_type == None:
        return True

    elif eval_type == 'R2':
        if r2 >= r2_min:
            return True
        else:
            return False

    elif eval_type == 'RMSE':
        if rmse <= rmse_min*y_bar:
            return True
        else:
            return False

    elif eval_type == 'Either':
        if r2 >= r2_min or rmse <= rmse_min*y_bar:
            return True
        else:
            return False

    elif eval_type == 'Both':
        if r2 >= r2_min and rmse <= rmse_min*y_bar:
            return True
        else:
            return False

    else:
        raise NameError('Invalid eval_type!')


def calculate_kH(ads_data, gas, eval_type='R2', r2_min=0.99, rmse_min=0.10, weight_type='error', fixed_intercept=False):

    comps = ads_data[gas+'_comp']
    mass = ads_data[gas+'_mass']
    error = ads_data[gas+'_error']

    offset = 0
    for i in range(len(comps)-offset,0,-1):

        # Check if enough points for a proper fit
        # Need to adjust this to stop before all points have same x value. Okay for now though.
        if i <= 6:
            return [None, None], None, None, None, None

        weights = create_weight_matrix(error[0:i], weight_type=weight_type)
        if fixed_intercept == False:
            p = np.polyfit(comps[0:i], mass[0:i], 1, cov=False, w=weights)
        if fixed_intercept == True:
            x = np.array(comps[0:i])[:,np.newaxis] * np.sqrt(weights[:,np.newaxis])
            y = np.array(mass[0:i]) * np.sqrt(weights)
            a, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
            p = [a, 0]
        r2, rmse, ybar = calculate_r2_and_rmse(p, comps[0:i], mass[0:i])
        calculate_r2_and_rmse(p, comps, mass)
        if check_r2_and_rmse(r2, r2_min, rmse, rmse_min, ybar, eval_type=eval_type) == False:
            offset += 1
        else:
            break

    max_comp = comps[i-1]
    return p, max_comp, r2, rmse, i


def calculate_kH_air(ads_data, gas, i, weight_type='error'):

    comps = ads_data[gas+'_comp']
    mass = ads_data['O2_mass']+ads_data['N2_mass']
    error = ads_data['O2_error']+ads_data['N2_error']

    weights = create_weight_matrix(error[0:i], weight_type=weight_type)
    p = np.polyfit(comps[0:i], mass[0:i], 1, cov=False, w=weights)
    r2, rmse, ybar = calculate_r2_and_rmse(p, comps[0:i], mass[0:i])

    return p, r2, rmse


def execute_henrys_coefficient_analysis(config_file):
    data = yaml_loader(config_file)
    gases = data['gases']
    ads_data_directory = data['ads_data_directory']
    results_directory = data['results_directory']

    columns = ['MOF', 'gas_kh', 'gas_r2', 'gas_rmse', 'air_kh', 'air_r2', 'air_rmse', 'kh', 'pure_air_mass', 'max_comp']

    for gas in gases:
        df = pd.DataFrame(columns=columns)
        all_files = list(glob.glob(ads_data_directory+gas+'/*.csv'))

        for file in all_files:
            ads_data = import_simulated_data(file)
            mof = ads_data['MOF'][0]
            p, max_comp, r2, rmse, i = calculate_kH(ads_data, gas, eval_type='R2', r2_min=0.99, weight_type='error', fixed_intercept=True)

            if i != None:
                p_air, r2_air, rmse_air = calculate_kH_air(ads_data, gas, i, weight_type='error')
                row = [mof, float(p[0]), r2, rmse, float(p_air[0]), r2_air, rmse_air, float(p[0]+p_air[0]), float(p_air[1]), max_comp]
            else:
                row = [mof, p[0], r2, rmse, None, None, None, None, None, max_comp]

            df = df.append(dict(zip(df.columns, row)), ignore_index=True)

        df.to_csv(results_directory+gas+'.csv', sep='\t', index=False)
