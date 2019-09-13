#!/usr/bin/env python3

# ----------------------------------
# ----- Import Python Packages -----
# ----------------------------------
import csv
import os
import sys
from datetime import datetime

import numpy as np
import sjs

from jobserver_utils import generate_unique_run_name
from sensor_array_mof_adsorption import read_composition_configuration, run_composition_simulation
from sensor_array_mof_adsorption import read_gases_configuration, read_mof_configuration_csv

# ----------------------------
# ----- System Arguments -----
# ----------------------------
mofs_filepath = sys.argv[1]
gas_comps_filepath = sys.argv[2]
gases_filepath = sys.argv[3]
pressure = sys.argv[4]

mofs, unit_cells = read_mof_configuration_csv(mofs_filepath)
compositions = read_composition_configuration(gas_comps_filepath)
gases = read_gases_configuration(gases_filepath)

# --------------------------
# ----- Some sjs stuff -----
# --------------------------
sjs.load(os.path.join("settings","sjs.yaml"))
job_queue = sjs.get_job_queue()

# ---------------------------------
# ----- Add jobs to job queue -----
# ---------------------------------
if job_queue is not None:
    print("Queueing jobs onto queue: %s" % job_queue)
    run_id_number = 0
    for i in range(len(mofs)):
        # --- Find the correspinging mof and unit cells ---
        mof = mofs[i]
        unit_cell = unit_cells[i]
        # --- Generate unique output directory ---
        run_name = generate_unique_run_name()
        output_dir = 'output_' + mof + '_%s' % run_name
        os.makedirs(output_dir)
        # ----- Setup CSV file and write header -----
        f = open(os.path.join(output_dir, mof+'.csv'),'w',newline='')
        header = ['Run ID','MOF','Mass']
        for gas in gases:
            header.append(gas)
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        # --- Queue info for simulating ---
        for composition in compositions:
            run_id = run_id_number
            job_queue.enqueue(run_composition_simulation, run_id, mof, unit_cell, pressure, gases, composition, csv_writer=None, output_dir=output_dir)
            run_id_number += 1
else:
    print("No job queue is setup. Running in serial mode here rather than on the cluster")
    for i in range(len(mofs)):
        # --- Find the correspinging mof and unit cells ---
        mof = mofs[i]
        unit_cell = unit_cells[i]
        # --- Generate unique output directory ---
        run_name = generate_unique_run_name()
        output_dir = 'output_' + mof + '_%s' % run_name
        os.makedirs(output_dir)
        # ----- Setup CSV file and write header -----
        f = open(os.path.join(output_dir, mof+'.csv'),'w',newline='')
        header = ['Run ID','MOF','Mass']
        for gas in gases:
            header.append(gas)
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        # --- Queue info for simulating ---
        for composition in compositions:
            run_id = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
            run_composition_simulation(run_id, mof, unit_cell, pressure, gases, composition, csv_writer=writer, output_dir=output_dir)

f.close()
