import ast
import csv
import glob
import matplotlib
import numpy as np
import os
import yaml
import matplotlib as mpl
from matplotlib import pyplot as plt


def yaml_loader(filepath):
    with open(filepath, 'r') as yaml_file:
        data = yaml.load(yaml_file)
    return data


def import_simulated_data(sim_results, sort_by_gas=False, gas_for_sorting=None):
    with open(sim_results) as file:
        reader = csv.DictReader(file, delimiter='\t')
        reader_list = list(reader)
        keys = reader.fieldnames

        for row in reader_list:
            # Isolate Mass Data since currently being assigned to single key
            mass_data_temp = [float(val) for val in row[keys[2]].split(' ')]
            num_gases = len(row)-len(mass_data_temp)-2
            # Reassign Compositions
            for i in range(num_gases):
                row[keys[-num_gases+i]] = row[keys[i+3]]
            # Reassign Masses
            for i in range(num_gases*2+2):
                row[keys[i+2]] = mass_data_temp[i]

        if sort_by_gas == True:
            reader_list = sorted(reader_list, key=lambda k: k[gas_for_sorting+'_comp'], reverse=False)

        return keys, reader_list




# ----- Calculating Herny's Coefficients -----
def plot_adsorbed_masses(all_comps, all_masses, all_errors, gases, figname=None, gas_name=None, mof_name=None):
    plt.clf()
    plt.figure(figsize=(4,3), dpi=600)

    for i in range(len(gases)):
        comps = all_comps[i]
        mass = all_masses[i]
        plt.plot(comps, mass, 'o', markersize=2, alpha=0.7)

    if mof_name != None:
        plt.title(mof_name)
    plt.xlabel(gases[0]+' Mole Fractions')
    plt.ylabel('Adsorbed Mass')
    plt.legend(gases)
    plt.tight_layout()
    if figname != None:
        plt.savefig(figname)
    plt.close()


def plot_kH_single_gas(comps, mass, error, max_comp, kH, intercept, R2, figname='None', gas_name='None', mof_name='None'):

    # Initialize Figure
    plt.clf()
    plt.figure(figsize=(4,3), dpi=600)
    if kH != None:
        fit = np.poly1d([kH, intercept])
        plt.plot(comps, fit(comps), 'r-')
    plt.errorbar(comps, mass, error, marker='o', markersize=3, elinewidth=1, linewidth=0)
    if mof_name != 'None':
        plt.title(mof_name, fontsize=12)
    if gas_name != 'None':
        plt.xlabel(gas+' Mole Fraction', fontsize=10)
    else:
        plt.xlabel('Mole Fraction', fontsize=10)
    plt.xticks(np.linspace(0,0.05,6), fontsize=8)
    plt.ylabel('Adsorbed Mass\n[mg/g Framework]', fontsize=10)
    plt.yticks(fontsize=8)
    plt.ylim([0.75*(min(mass)-max(error)),1.25*(max(mass)+max(error))])
    plt.tight_layout()
    textstr='\n'.join(['K_H = '+str(np.round(kH, 2)),'Max. Comp. = '+str(np.round(max_comp, 3))])
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    plt.text(0.3*max(comps), 1.08*max(mass)+max(error), textstr, fontsize=10, bbox=props)
    plt.savefig(figname)
    plt.close()


def plot_kH_air(comps, mass, error, max_comp, kH, intercept, R2, figname='None', gas_name='None', mof_name='None'):
    # Redefine the Fit
    fit = np.poly1d([kH, intercept])

    # Initialize Figure
    plt.clf()
    plt.figure(figsize=(4,3), dpi=600)
    plt.plot(comps, fit(comps), 'r-')
    plt.errorbar(comps, mass, error, marker='o', markersize=3, elinewidth=1, linewidth=0)
    if mof_name != 'None':
        plt.title(mof_name, fontsize=12)
    if gas_name != 'None':
        plt.xlabel(gas+' Mole Fraction', fontsize=10)
    else:
        plt.xlabel('Mole Fraction', fontsize=10)
    # plt.xticks(np.linspace(0,0.05,6), fontsize=8)
    plt.ylabel('Adsorbed Mass\n[mg/g Framework]', fontsize=10)
    plt.yticks(fontsize=8)
    plt.ylim([0,1.5*(max(mass)+max(error))])
    plt.tight_layout()
    textstr='\n'.join(['K_H = '+str(np.round(kH, 2)),'Max. Comp. = '+str(np.round(max_comp, 3))])
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    plt.text(0.01, 1.4*(max(mass)+max(error)), textstr, fontsize=10, verticalalignment='top', bbox=props)
    plt.savefig(figname)
    plt.close()


def plot_kH_multiple_gases(all_comps, all_masses, all_errors, kH, intercept, figname='None', gas_name='None', mof_name='None'):
    # Define Colors
    cmap_vals = np.linspace(0,1,2)
    colors = [mpl.cm.bwr(x) for x in cmap_vals]

    # Initialize Figure
    plt.clf()
    plt.figure(figsize=(4,3), dpi=600)

    for i in range(all_comps):
        kH = all_kHs[i]
        comps = all_comps[i]
        mass = all_masses[i]
        error = all_errors[i]
        intercepts = all_intercepts[i]

        fit = np.poly1d([kH, intercept])
        plt.plot(comps, fit(comps), 'r-')
        plt.errorbar(comps, mass, error, marker='o', markersize=3, elinewidth=1, linewidth=0, color=colors[0])

    # Define labels with respect to Henry's Gas (positionally first).
    if mof_name != 'None':
        plt.title(mof_name, fontsize=12)
    if gas_name != 'None':
        plt.xlabel(gas+' Mole Fraction', fontsize=10)
    else:
        plt.xlabel('Mole Fraction', fontsize=10)
    plt.xticks(np.linspace(0,0.05,6), fontsize=8)
    plt.ylabel('Absolute Loading\n[mg/g Framework]', fontsize=10)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()


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


def analyze_single_ratio(gas, air_ratio, sim_results_path, analysis_results_path, force_intercept_hg=True, weight='error', r2_min=0.99):
    """
    Acutally Calculating Henry's Coefficients with Error
    Each air ratio analyzed/plotted separately
    """

    kH_results = []
    for file in list(glob.glob(sim_results_path+'*/*.csv')):
        # Prepare Dict
        temp_dict = {}
        temp_dict['Gas'] = gas
        temp_dict['Background'] = air_ratio

        # Load Results / Extract MOF Name
        keys, dict_data = import_simulated_data(file, gas)
        mof_name = dict_data[0]['MOF']
        temp_dict['MOF'] = mof_name

        # Isolate Data for Fitting
        comps = [float(row[gas+'_comp']) for row in dict_data]
        mass = [float(row[gas+'_mass']) for row in dict_data]
        error = [float(row[gas+'_error']) for row in dict_data]
        total_mass = [float(row['total_mass']) for row in dict_data]
        total_error = [float(row['total_mass_error']) for row in dict_data]

        # Calculate kH (uncomment desired method)
        figname = analysis_results_path+mof_name+'_kH.png'
        if force_intercept_hg == True:
            kH, R2, max_comp = calculate_kH_fixed_intercept(comps, mass, error, r2_min=r2_min, weight=weight, figname=figname, gas_name=gas, mof_name=mof_name)
        elif force_intercept_hg == False:
            kH, intercept, R2, max_comp = calculate_kH(comps, mass, error, r2_min=r2_min, weight=weight, figname=figname, gas_name=gas, mof_name=mof_name)
        else:
            print('Invalid Intercept Option!')


        temp_dict['k_H'] = kH[0]
        temp_dict['R^2'] = R2
        temp_dict['Maximum Composition'] = max_comp
        kH_results.extend([temp_dict])

    filename = '/Users/brian_day/Desktop/HC_Work/HenrysConstants_Analysis_Results/'+folder+'_henrys_coefficients.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in kH_results:
            writer.writerow([row])


def analyze_all_ratios(gas, sim_results_path, analysis_results_path, force_intercept_hg=True, weight='error', hg_eval_type='R2', air_eval_type='Either', r2_min_hg=0.99, r2_min_air=0.99, rmse_min_hg=0.10, rmse_min_air=0.10, consolidate_data=False):
    """
    Acutally Calculating Henry's Coefficients with Error
    All Air Ratios Combined into Single Data Set
    """

    all_files = list(glob.glob(sim_results_path+str(gas)+'*/*/*.csv'))
    # all_files = list(glob.glob(sim_results_path+'*/*.csv'))
    all_files.sort()

    kH_results_hg = []
    kH_results_air = []
    for file in all_files[0:50]:
        # Establish counter to use when eliminating datapoints
        counter = 1

        # Extract Filename
        filename = file.split('/')[-1]

        # Prepare Dict
        temp_dict_hg = {}
        temp_dict_air = {}
        temp_dict_hg['Gas'] = gas
        temp_dict_air['Gas'] = gas

        # Load Results / Extract MOF Name
        keys, dict_data = import_simulated_data(file, gas)
        mof_name = dict_data[0]['MOF']
        temp_dict_hg['MOF'] = mof_name
        temp_dict_air['MOF'] = mof_name

        # Isolate Data for Fitting
        # Henry's Gas
        comps = [float(row[gas+'_comp']) for row in dict_data]
        mass = [float(row[gas+'_mass']) for row in dict_data]
        error = [float(row[gas+'_error']) for row in dict_data]
        # Nitrogen
        comps_N2 = [float(row['N2_comp']) for row in dict_data]
        mass_N2 = [float(row['N2_mass']) for row in dict_data]
        error_N2 = [float(row['N2_error']) for row in dict_data]
        # Oxygen
        comps_O2 = [float(row['O2_comp']) for row in dict_data]
        mass_O2 = [float(row['O2_mass']) for row in dict_data]
        error_O2 = [float(row['O2_error']) for row in dict_data]
        # Air: Nitrogen + Oxygen
        comps_air = [sum(x) for x in zip(comps_N2, comps_O2)]
        mass_air = [sum(x) for x in zip(mass_N2, mass_O2)]
        error_air = [sum(x) for x in zip(error_N2, error_O2)]
        # Total Mass
        total_mass = [float(row['total_mass']) for row in dict_data]
        total_error = [float(row['total_mass_error']) for row in dict_data]

        for file2 in all_files[50::]:

            # Check if same MOF
            filename2 = file2.split('/')[-1]
            if filename2 == filename:
                # Update counter
                counter += 1

                # Load Results
                keys, dict_data = import_simulated_data(file2, gas)

                # Isolate Data for Fitting
                comps_temp = [float(row[gas+'_comp']) for row in dict_data]
                mass_temp = [float(row[gas+'_mass']) for row in dict_data]
                error_temp = [float(row[gas+'_error']) for row in dict_data]
                # Nitrogen
                comps_N2_temp = [float(row['N2_comp']) for row in dict_data]
                mass_N2_temp = [float(row['N2_mass']) for row in dict_data]
                error_N2_temp = [float(row['N2_error']) for row in dict_data]
                # Oxygen
                comps_O2_temp = [float(row['O2_comp']) for row in dict_data]
                mass_O2_temp = [float(row['O2_mass']) for row in dict_data]
                error_O2_temp = [float(row['O2_error']) for row in dict_data]
                # Air: Nitrogen + Oxygen
                comps_air_temp = [sum(x) for x in zip(comps_N2_temp, comps_O2_temp)]
                mass_air_temp = [sum(x) for x in zip(mass_N2_temp, mass_O2_temp)]
                error_air_temp = [sum(x) for x in zip(error_N2_temp, error_O2_temp)]
                # Total Mass
                total_mass_temp = [float(row['total_mass']) for row in dict_data]
                total_error_temp = [float(row['total_mass_error']) for row in dict_data]

                # Add Data to Original Data Set
                # Henry's Gas
                comps.extend(comps_temp)
                mass.extend(mass_temp)
                error.extend(error_temp)
                # Nitrogen
                comps_N2.extend(comps_N2_temp)
                mass_N2.extend(mass_N2_temp)
                error_N2.extend(error_N2_temp)
                # Oxygen
                comps_O2.extend(comps_O2_temp)
                mass_O2.extend(mass_O2_temp)
                error_O2.extend(error_O2_temp)
                # Air: Nitrogen + Oxygen
                comps_air.extend(comps_air_temp)
                mass_air.extend(mass_air_temp)
                error_air.extend(error_air_temp)
                # Total Mass
                total_mass.extend(total_mass_temp)
                total_error.extend(total_error_temp)

        # Write Raw Air Data (sorted) to a new file for inspection
        if consolidate_data == True:
            indicies = np.argsort(comps)
            comps_sorted = [comps_air[i] for i in indicies]
            mass_sorted = [mass_air[i] for i in indicies]
            error_sorted = [error_air[i] for i in indicies]
            all_data = np.vstack([comps_sorted, mass_sorted, error_sorted]).T
            header = ['Comps', 'Mass', 'Error']

            filepath = analysis_results_path+'rawdata/'
            filename = filepath+mof_name+'_airdata.csv'

            if os.path.isdir(filepath) != True:
                os.mkdir(filepath)

            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                # writer.writerows([comps_sorted])
                # writer.writerows([mass_sorted])
                # writer.writerows([error_sorted])
                writer.writerow(header)
                for row in all_data:
                    writer.writerow(row)

        # Plot Adsorption Data
        figname = analysis_results_path+mof_name+'_ads_data.png'
        all_comps = [comps, comps, comps]
        all_masses = [mass, mass_air, total_mass]
        all_errors = [error, error_air, total_error]
        gases = [gas, 'Air', 'Total']
        plot_adsorbed_masses(all_comps, all_masses, all_errors, gases, figname=figname, gas_name=gas, mof_name=mof_name)

        # Calculate kH - Henry's Gas
        if force_intercept_hg == True:
            kH, max_comp, R2, RMSE, i_offset = calculate_kH_fixed_intercept(comps, mass, error, eval_type=hg_eval_type, r2_min=r2_min_hg, rmse_min=rmse_min_hg, weight=weight, counter=counter)
            intercept = 0
        elif force_intercept_hg == False:
            kH, intercept, max_comp, R2, RMSE, i_offset = calculate_kH(comps, mass, error, eval_type=hg_eval_type, r2_min=r2_min_hg, rmse_min=rmse_min_hg, weight=weight, counter=counter)
        else:
            print('Invalid Intercept Option!')

        if kH != None:
            figname = analysis_results_path+mof_name+'_kH.png'
            plot_kH_single_gas(comps, mass, error, max_comp, float(kH), intercept, R2, figname=figname, gas_name=gas, mof_name=mof_name)

        temp_dict_hg['k_H'] = kH
        # temp_dict_hg['Intercept'] = intercept
        temp_dict_hg['R^2'] = R2
        temp_dict_hg['RMSE'] = RMSE
        temp_dict_hg['Maximum Composition'] = max_comp
        kH_results_hg.extend([temp_dict_hg])

        # Calculate kH - Air Mixture
        if air_eval_type == 'With hg':
            if kH != None:
                kH, intercept, max_comp, R2, RMSE, _ = calculate_kH(comps_air, mass_air, error_air, eval_type=None, r2_min=r2_min_air, rmse_min=rmse_min_air, weight=weight, flipped=True, i_offset=i_offset, counter=counter)
            else:
                kH, intercept, max_comp, R2, RMSE, _ = (None, 0, 0, 0, 0, 0)
        else:
            kH, intercept, max_comp, R2, RMSE, _ = calculate_kH(comps_air, mass_air, error_air, eval_type=air_eval_type, r2_min=r2_min_air, rmse_min=rmse_min_air, weight=weight, flipped=True, counter=counter)

        if kH != None:
            # figname = analysis_results_path+mof_name+'_kH_air.png'
            # plot_kH_air(comps_air, mass_air, error_air, max_comp, float(kH), intercept, R2, figname=figname, gas_name=gas, mof_name=mof_name)
            figname = analysis_results_path+mof_name+'_kH_air_on_henry.png'
            plot_kH_single_gas(comps, mass_air, error_air, max_comp, -float(kH), intercept+kH, R2, figname=figname, gas_name=gas, mof_name=mof_name)

        temp_dict_air['k_H'] = kH
        if kH != None:
            temp_dict_air['Pure Air Mass'] = kH+intercept
        else:
            temp_dict_air['Pure Air Mass'] = None
        temp_dict_air['R^2'] = R2
        temp_dict_air['RMSE'] = RMSE
        temp_dict_air['Maximum Composition'] = max_comp
        kH_results_air.extend([temp_dict_air])

    # Write Henry's Gas csv
    filename = '/Users/brian_day/Desktop/HC_Work/HenrysConstants_Analysis_Results/'+str(gas)+'_AllRatios/_henrys_coefficients_hg.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in kH_results_hg:
            writer.writerow([row])

    # Write Air csv
    filename = '/Users/brian_day/Desktop/HC_Work/HenrysConstants_Analysis_Results/'+str(gas)+'_AllRatios/_henrys_coefficients_air.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in kH_results_air:
            writer.writerow([row])


def read_kH_results(filename):
    with open(filename, newline='') as csvfile:
        output_data = csv.reader(csvfile, delimiter="\t")
        output_data = list(output_data)
        full_array = []
        for i in range(len(output_data)):
            row = output_data[i][0]
            row = row.replace('nan', '\'nan\'')
            row = row.replace('inf', '\'inf\'')
            row = row.replace('-\'inf\'', '\'-inf\'')
            temp_array = []
            temp_row = ast.literal_eval(row)
            # if type(temp_row['R^2']) == str or temp_row['R^2'] < 0:
            #     continue
            temp_array.append(temp_row)
            full_array.extend(temp_array)
        return full_array


def plot_all_kH(gas, data, figname):
    colors = mpl.cm.get_cmap('RdBu')
    color0 = colors(0.80)
    color1 = colors(0.20)

    comps = []
    khs = []
    for row in data:
        comps.extend([row['Maximum Composition']])
        khs.extend([row['k_H']])

    if gas != 'None':
        if gas == 'CO2':
            gas_for_title = '$CO_2$'
        else:
            gas_for_title = '$' + gas[0].upper() + gas[1::] +'$'

    plt.clf()
    plt.figure(figsize=(4.5,4.5), dpi=600)
    plt.semilogy(comps,khs,'o', alpha=0.7, color=color0)
    plt.ylim([1e-2,1e8])
    plt.xlim([-0.001,0.051])
    plt.title(gas_for_title+' in Air', fontsize=16)
    plt.xlabel('Maximum Mole Fraction\n(End of Henry\'s Regime)', fontsize=16)
    plt.ylabel('Henry\'s Coefficient\n[mg/g/mole fraction]',fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()
