#!/usr/bin/env python3

# ----------------------------------
# ----- Import Python Packages -----
# ----------------------------------
import csv
from datetime import datetime
import numpy as np
import os
import shutil
import sjs
import subprocess
import sys
from textwrap import dedent
import yaml

# -----------------------------------------
# ----- User Defined Python Functions -----
# -----------------------------------------
def yaml_loader(filepath):
    with open(filepath, 'r') as yaml_file:
        data = yaml.load(yaml_file)
    return(data)


def generate_unique_run_name():
    return datetime.now().strftime("%Y_%m_%d__%H_%M_%S")


def generate_unique_per_process_filename():
    return "%s_%s" % (os.uname()[1], os.getpid())


def read_mof_configuration_csv(filename):
    openfile = open(filename, 'rt')
    mofs_uc_csv = csv.reader(openfile, delimiter='\t')
    mofs_uc = []
    for i in mofs_uc_csv:
        mofs_uc.append(i)
    openfile.close()
    mofs = []
    unit_cells = []
    for i in range(len(mofs_uc)):
        mofs.append(mofs_uc[i][0])
        unit_cells.append(" ".join(mofs_uc[i][1:]))
    return mofs, unit_cells


def read_gases_configuration(filename):
    with open(filename) as f:
        return [ line.strip() for line in f.readlines() ]


def read_composition_configuration(filename):
    with open(filename,newline='') as csvfile:
        comp_reader = csv.DictReader(csvfile, delimiter="\t")
        return list(comp_reader)


def read_pressure_configuration(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        pressures = [float(row[0]) for row in reader]
        return pressures


def write_raspa_file(filename, mof, unit_cell, pressure, gases, composition, config_file):
    config_data = yaml_loader(config_file)
    gas_names_def = config_data['Forcefield_Gas_Names']
    f = open(filename,'w',newline='')

    simulation_file_header = """\
	SimulationType                MonteCarlo
	NumberOfCycles                2000
	NumberOfInitializationCycles  1000
	PrintEvery                    200

	ChargeMethod                  Ewald
	CutOff                        12.0
	Forcefield                    JennaUFF2
	EwaldPrecision                1e-6

	Framework 0
	FrameworkName %s
	UnitCells %s
	HeliumVoidFraction 0.81
	UseChargesFromCIFFile yes
	ExternalTemperature 298.0
	ExternalPressure %s
	""" % (mof, unit_cell, pressure)

    f.write(dedent(simulation_file_header))

    component_number = 0
    for gas in gases:
        gas_name = gas_names_def[gas]
        mole_fraction = composition[gas]
        simulation_file_gas = """
    Component %s MoleculeName              %s
                 MoleculeDefinition         TraPPE-Zhang
                 MolFraction                %s
                 TranslationProbability     0.5
                 RegrowProbability          0.5
                 IdentityChangeProbability  1.0
                   NumberOfIdentityChanges  2
                   IdentityChangesList      0 1
                 SwapProbability            1.0
                 CreateNumberOfMolecules    0

                 """ % (component_number, gas_name, mole_fraction)

        f.write(dedent(simulation_file_gas))
        component_number += 1

    f.close()


def parse_output(output_file):
    mass = float(subprocess.check_output(['./calculate_mass.sh', output_file]))
    return mass


def run(run_id, mof, unit_cell, pressure, gases, composition, config_file, output_dir='output'):
    # create unique working directory for this simulation
    working_dir = os.path.join(output_dir, generate_unique_per_process_filename())
    os.makedirs(working_dir, exist_ok=True)

    # run simulation
    write_raspa_file(os.path.join(working_dir, "simulation.input"), mof, unit_cell, pressure, gases,composition, config_file)
    subprocess.run(['simulate', 'simulation.input'], check=True, cwd=working_dir)

    # parse data from simulation
    data_filename = os.path.join(working_dir, 'Output', 'System_0', '*.data')
    mass = parse_output(data_filename)

    # archive data and configuration; delete working_dir
    run_descriptor = "%s" % (run_id)
    archive_dir = os.path.join(output_dir, 'archive', run_descriptor)
    os.makedirs(archive_dir, exist_ok=True)
    for f in ["Output", "simulation.input"]:
        shutil.move(os.path.join(working_dir, f), archive_dir)

    shutil.rmtree(os.path.join(working_dir))

    return (mass)


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
    mass = run(run_id, mof, unit_cell, pressure, gases, composition, 'config_files/write_comps_config.yaml', output_dir=output_dir)
    csv_writer.writerow([run_id, mof, mass, *[composition[gas] for gas in gases]])
    # ----- Close the file, if we opened it above -----
    if csv_file is not None:
        csv_file.close()

# --------------------------
# ----- Some sjs stuff -----
# --------------------------
sjs.load(os.path.join("settings","sjs.yaml"))
job_queue = sjs.get_job_queue()

def execute_write_simulations(config_file):

    data = yaml_loader(config_file)
    mofs_filepath = data['mofs_filepath']
    compositions_filepath = data['comps_filepath']
    gases_filepath = data['gases_filepath']
    pressures_filepath = data['pressure']

    mofs, unit_cells = read_mof_configuration_csv(mofs_filepath)
    compositions = read_composition_configuration(compositions_filepath)
    gases = read_gases_configuration(gases_filepath)
    pressures = read_pressure_configuration(pressures_filepath)

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
            for pressure in pressures:
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
            for pressure in pressures:
                for composition in compositions:
                    run_id = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
                    run_composition_simulation(run_id, mof, unit_cell, pressure, gases, composition, csv_writer=writer, output_dir=output_dir)

    f.close()
