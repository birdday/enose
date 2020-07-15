# --------------------------------------------
# ----- Import Python Commands/Functions -----
# --------------------------------------------
import csv
import os

import sensor_array_mof_adsorption_simulation as sim
from jobserver_utils import generate_unique_per_process_filename

# -----------------------------------------
# ----- User Defined Python Functions -----
# -----------------------------------------
def read_gases_configuration(filename):
    with open(filename) as f:
        return [ line.strip() for line in f.readlines() ]

def read_mof_configuration_csv(filename):
    openfile = open(filename, 'rt')
    mofs_uc_csv = csv.reader(openfile, delimiter='\t')
    mofs_uc = []
    for i in mofs_uc_csv:
        mofs_uc.append(i)
    openfile.close()
    s = " "
    mofs = []
    unit_cells = []
    for i in range(len(mofs_uc)):
        mofs.append(mofs_uc[i][0])
        unit_cells.append(s.join(mofs_uc[i][1:]))
    return mofs, unit_cells

def read_composition_configuration(filename):
    with open(filename,newline='') as csvfile:
        comp_reader = csv.DictReader(csvfile, delimiter="\t")
        return list(comp_reader)


def read_pressure_configuration(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        pressures = [float(row[0]) for row in reader]
        return pressures

def run_composition_simulation(run_id, mof, pressure, gases, composition, csv_writer=None, output_dir='output'):
    # ----- If there is no csv_writer passed, we write to a file that is unique to this process -----
    csv_file = None
    if csv_writer is None:
        results_dir = os.path.join(output_dir,'results')
        os.makedirs(results_dir, exist_ok=True)
        filename = os.path.join(results_dir, generate_unique_per_process_filename() + ".csv")
        csv_file = open(filename,'a',newline='')
        csv_writer = csv.writer(csv_file, delimiter='\t')
    # ----- Run the simulation / Output the data -----
    mass = sim.run(run_id, mof, unit_cell, pressure, gases, composition, 'config_files/write_comps_config.yaml', output_dir=output_dir)
    csv_writer.writerow([run_id, mof, mass, *[composition[gas] for gas in gases]])
    # ----- Close the file, if we opened it above -----
    if csv_file is not None:
        csv_file.close()
