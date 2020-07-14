# Written by Brian Day
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# ----- RASPA Data -----
# Break the file down into component parts, and then rejoin. This is just to
# make sure the format is right for additional processing.
results_filepath = 'CO2_0p1bar/CO2_Sensing_Results'
gases = ['CO2','O2','N2']

# Make cleaned up, tab-delimited files
results_newdir = results_filepath+'_New'
os.mkdir(results_newdir)
col_names = np.array(['Run_ID','MOF','Mass_Uptake','CO2','O2','N2'])
allmof_results = np.array(['Run_ID','MOF','Mass_Uptake','CO2','O2','N2'])

# Create list of compositions missing for each MOF
missingcomps_dir = 'CO2_0p1bar/CO2_Sensing_MissingComps'
os.mkdir(missingcomps_dir)
run_id_list = np.linspace(0,960,num=961).astype(int)
run_id_list.tolist()
comp_list = pd.read_csv('comps_1%.csv', delimiter='\t|,|:|;', engine='python')
comp_list = comp_list.values

for filename in os.listdir(results_filepath):
    print(filename)

    if filename.endswith('.csv'):
        ads_file = results_filepath+'/'+filename
        ads_results = pd.read_csv(ads_file, delimiter='\t|,|:|;', engine='python', header=None, names=col_names)
        ads_results = ads_results.drop(index=0)

        # Remove extraneous characters
        ads_results = ads_results.replace({'_eqeq':''}, regex=True)
        ads_results = ads_results.replace({'_EQeq':''}, regex=True)
        ads_results = ads_results.replace({'_EQEQ':''}, regex=True)
        ads_results = ads_results.replace({'_v2':''}, regex=True)
        ads_results = ads_results.replace({'\[':''}, regex=True)
        ads_results = ads_results.replace({'\]':''}, regex=True)
        ads_results = ads_results.replace({'\'':''}, regex=True)

        # Convert to Numpy array
        ads_results = ads_results.values

        # Manually fix the data types
        ads_results[:,0] = ads_results[:,0].astype(int)
        ads_results[:,1] = ads_results[:,1].astype(str)
        ads_results[:,2:6] = ads_results[:,2:6].astype(float)

        # Sort according to Run_ID
        ads_results = ads_results[ads_results[:,0].argsort()]

        # Name the new results file
        mof_name = ads_results[1,1]
        new_filename = mof_name+'.csv'
        new_filename = new_filename.replace('_eqeq','')
        new_filename = new_filename.replace('_v2_EQeq','') # Needed for HKUST-1

        # Adjust the Run_ids (quick fix for missing comps)
        for i in range(len(ads_results[:,0])):
            for j in range(len(comp_list)):
                if np.array_equal(ads_results[i,3:6],comp_list[j]) == True:
                    ads_results[i,0] = int(run_id_list[j])

        # Sort again according to new Run_ID
        ads_results = ads_results[ads_results[:,0].argsort()]

        # Check for and delete duplicate simulations (first one is only ever kept)
        rows_to_delete = []
        for i in range(1, len(ads_results[:,0])):
            if ads_results[i,0] == ads_results[i-1,0]:
                rows_to_delete.append(i)

        ads_results = np.delete(ads_results, rows_to_delete, axis=0)

        # Append results to allmof-results files (before adding column header)
        allmof_results = np.vstack([allmof_results, ads_results]);

        # Add column headers
        ads_results = np.vstack([col_names, ads_results])

        # Write the new results file
        with open(new_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(ads_results)

        # Use rename to move file into correct directory
        os.rename(new_filename, results_newdir+'/'+new_filename)

        # Check for missing compositions by run_id
        comps_filename=mof_name+'_comps.csv'
        ads_results_id_list = ads_results[:,0].tolist()
        ads_results_comp_list = ads_results[:,3:6]
        missing_run_ids = list(set(run_id_list).difference(ads_results_id_list))

        # Write the comps files for follow-up runs
        if len(missing_run_ids) != 0:
            missing_comps = comp_list[missing_run_ids]
            with open(comps_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(gases)
                for each_line in missing_comps:
                        writer.writerow(each_line)

            # Use rename to move file into correct directory
            os.rename(comps_filename, missingcomps_dir+'/'+comps_filename)

# Write the allmof-results file
with open('ALL_MOFS.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerows(allmof_results)

os.rename('ALL_MOFS.csv', results_newdir+'/'+'ALL_MOFS.csv')
