import glob
import numpy as np
import os
import yaml
import pandas as pd


def yaml_loader(filepath):
    with open(filepath, 'r') as yaml_file:
        data = yaml.load(yaml_file)
    return data


def import_simulated_data(filename):
    ads_data = pd.read_csv(filename, sep='\t', engine='python')
    return ads_data





def calculate_kH(comps, mass, error, eval_type='R2', r2_min=0.99, rmse_min=0.10, weight='error', flipped=False, i_offset=0, counter=1):

    # Make sure data is properly sorted
    indicies = np.argsort(comps)
    comps_sorted = [comps[i] for i in indicies]
    mass_sorted = [mass[i] for i in indicies]
    error_sorted = [error[i] for i in indicies]
    comps = comps_sorted
    mass = mass_sorted
    error = error_sorted

    if flipped == True:
        print('Flipped!')
        comps = np.flipud(comps)
        mass = np.flipud(mass)
        error = np.flipud(error)

    for i in range(len(comps)-i_offset,-counter,-1):

        # Check if enough points for a proper fit
        if i <= 2:
            print('Could not fit to given data!')
            return None, 0, 0, 0, 0, 0

        # Create a weighting matrix
        if weight == 'None':
            weights=np.ones(len(error[0:i]))
        elif weight == 'error':
            if 0 not in error[0:i]:
                weights=np.divide(1,error[0:i])
            else:
                weights_temp = [1/val for val in error[0:i] if val != 0]
                if len(weights_temp) != 0:
                    weights = np.array([1/val if val != 0 else max(weights_temp) for val in error[0:i]])
                else:
                    weights = np.ones(len(error[0:i]))
        elif weight == 'error_squared':
            error_squared = []
            for j in error[0:i]:
                error_squared.extend([j**2])
            if 0 not in error_sqaured:
                weights = np.divide(1,error_squared)
            else:
                weights_temp = [1/val for val in error_squared if val != 0]
                if len(weights_temp) != 0:
                    weights = np.array([1/val if val != 0 else max(weights_temp) for val in error_squared])
                else:
                    weights = np.ones(len(error_sqaured[0:i]))
        else:
            raise NameError('Invalid Error Type!')

        # Fit a Line
        # p, V = np.polyfit(comps[0:i], mass[0:i], 1, cov=True, w=weights)
        p = np.polyfit(comps[0:i], mass[0:i], 1, cov=False, w=weights)

        # Calculate R_Squared and RMSE
        fit = np.poly1d(p)
        predicted_vals = fit(comps[0:i])
        residuals = predicted_vals-mass[0:i]
        ybar = np.sum(mass[0:i])/len(mass[0:i])
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((mass[0:i]-ybar)**2)
        R2 = 1-(ss_res/ss_tot)
        MSE = ss_res/len(mass[0:i])
        RMSE = MSE**(0.5)

        max_comp = comps[i-1]

        # Check R_Squared and RMSE
        if eval_type == None:
            break
        elif eval_type == 'R2':
            if R2 >= r2_min:
                break
        elif eval_type == 'RMSE':
            if RMSE <= rmse_min*ybar:
                break
        elif eval_type == 'Either':
            if R2 >= r2_min or RMSE <= rmse_min*ybar:
                break
        elif eval_type == 'Both':
            if R2 >= r2_min and RMSE <= rmse_min*ybar:
                break
        else:
            raise NameError('Invalid eval_type!')

    return p[0], p[1], max_comp, R2, RMSE, len(comps)-i


def calculate_kH_alt(comps, mass, error, r2_min=0.99, weight='error', flipped=False):

    # Make sure data is properly sorted
    indicies = np.argsort(comps)
    comps_sorted = [comps[i] for i in indicies]
    mass_sorted = [mass[i] for i in indicies]
    error_sorted = [error[i] for i in indicies]
    comps = comps_sorted
    mass = mass_sorted
    error = error_sorted

    if flipped == True:
        print('Flipped!')
        comps = np.flipud(comps)
        mass = np.flipud(mass)
        error = np.flipud(error)

    for i in range(len(comps),-1,-1):

        # Check if enough points for a proper fit
        if i <= 2:
            print('Could not fit to given data!')
            return None, 0, 0, 0

        # Linear Fit
        if weight == 'None':
            weights=np.ones(len(error[0:i]))
        elif weight == 'error':
            if 0 not in error[0:i]:
                weights=np.divide(1,error[0:i])
            else:
                weights_temp = [1/val for val in error[0:i] if val != 0]
                if len(weights_temp) != 0:
                    weights = np.array([1/val if val != 0 else max(weights_temp) for val in error[0:i]])
                else:
                    weights = np.ones(len(error[0:i]))
        elif weight == 'error_squared':
            error_squared = []
            for j in error[0:i]:
                error_squared.extend([j**2])
            if 0 not in error_sqaured:
                weights = np.divide(1,error_squared)
            else:
                weights_temp = [1/val for val in error_squared if val != 0]
                if len(weights_temp) != 0:
                    weights = np.array([1/val if val != 0 else max(weights_temp) for val in error_squared])
                else:
                    weights = np.ones(len(error_sqaured[0:i]))
        else:
            raise NameError('Invalid Error Type!')

        x = np.array(comps[0:i])
        x = np.vstack([x, np.ones(len(x))]).T
        y = np.array(mass[0:i])

        xw = x*np.sqrt(weights[:,np.newaxis])
        yw = y*np.sqrt(weights)
        a, b, c, d = np.linalg.lstsq(xw, yw, rcond=None)

        # Calculate R_Squared
        fit = np.poly1d(a)
        predicted_vals = fit(comps[0:i])
        residuals = predicted_vals-mass[0:i]
        ybar = np.sum(mass[0:i])/len(mass[0:i])
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((mass[0:i]-ybar)**2)
        R2 = 1-(ss_res/ss_tot)

        max_comp = comps[i-1]

        # Check R_Squared
        # print('i = '+str(i)+'\t R^2 = '+str(R2))
        if R2 >= r2_min:
            # print('Number of Points Used: ', len(comps[0:i]))
            # print('R^2 = ', R2)
            # print('Henry Coeff. = ', p[0])
            # print('Intercept (should be 0) = ',p[1])
            max_comp = comps[i-1]
            break

    return a[0], a[1], R2, max_comp


def calculate_kH_fixed_intercept(comps, mass, error, eval_type='R2', r2_min=0.99, rmse_min=0.10, weight='error', counter=1):

    # Make sure data is properly sorted
    indicies = np.argsort(comps)
    comps_sorted = [comps[i] for i in indicies]
    mass_sorted = [mass[i] for i in indicies]
    error_sorted = [error[i] for i in indicies]
    comps = comps_sorted
    mass = mass_sorted
    error = error_sorted

    for i in range(len(comps),-counter,-1):

        # Check if out of points
        if i <= 2:
            print('Could not fit to given data!')
            return None, 0, 0, 0, 0

        # Linear Fit
        if weight == 'None':
            weights=np.ones(len(error[0:i]))
        elif weight == 'error':
            if 0 not in error[0:i]:
                weights=np.divide(1,error[0:i])
            else:
                weights_temp = [1/val for val in error[0:i] if val != 0]
                if len(weights_temp) != 0:
                    weights = np.array([1/val if val != 0 else max(weights_temp) for val in error[0:i]])
                else:
                    weights = np.ones(len(error[0:i]))
        elif weight == 'error_squared':
            error_squared = []
            for j in error[0:i]:
                error_squared.extend([j**2])
            if 0 not in error_sqaured:
                weights = np.divide(1,error_squared)
            else:
                weights_temp = [1/val for val in error_squared if val != 0]
                if len(weights_temp) != 0:
                    weights = np.array([1/val if val != 0 else max(weights_temp) for val in error_squared])
                else:
                    weights = np.ones(len(error_sqaured[0:i]))
        else:
            raise NameError('Invalid Error Type!')

        x = np.array(comps[0:i])
        x = x[:,np.newaxis]
        y = np.array(mass[0:i])

        xw = x*np.sqrt(weights[:,np.newaxis])
        yw = y*np.sqrt(weights)
        a, b, c, d = np.linalg.lstsq(xw, yw, rcond=None)

        # Calculate R_Squared
        fit = np.poly1d([a[0], 0])
        predicted_vals = fit(comps[0:i])
        residuals = predicted_vals-mass[0:i]
        ybar = np.sum(mass[0:i])/len(mass[0:i])
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((mass[0:i]-ybar)**2)
        R2 = 1-(ss_res/ss_tot)
        MSE = ss_res/len(mass[0:i])
        RMSE = MSE**(0.5)

        max_comp = comps[i-1]

        # Check R_Squared and RMSE
        if eval_type == 'R2':
            if R2 >= r2_min:
                break
        elif eval_type == 'RMSE':
            if RMSE <= rmse_min*ybar:
                break
        elif eval_type == 'Either':
            if R2 >= r2_min or RMSE <= rmse_min*ybar:
                break
        elif eval_type == 'Both':
            if R2 >= r2_min and RMSE <= rmse_min*ybar:
                break
        else:
            raise NameError('Invalid eval_type!')

    return a[0], max_comp, R2, RMSE, len(comps)-i
