import csv
import os
import shutil
import subprocess
from datetime import datetime
from textwrap import dedent

import pandas as pd
import yaml

def yaml_loader(filepath):
    with open(filepath, 'r') as yaml_file:
        data = yaml.load(yaml_file)
    return(data)


def generate_unique_run_name():
    return datetime.now().strftime("%Y_%m_%d__%H_%M_%S")


def generate_unique_per_process_filename():
    return "%s_%s" % (os.uname()[1], os.getpid())


def read_mof_configuration_csv(filename):
    mofs_unitcells = pd.read_csv(filename, sep='\t', engine='python')
    mofs = mofs_unitcells['MOF_cif'].values
    unitcells = mofs_unitcells[['a','b','c']].values
    return mofs, unitcells


def read_composition_configuration(filename):
    return pd.read_csv(filename, sep='\t', engine='python')


def write_raspa_file(filename, mof, unitcell, composition, pressure):
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
	""" % (mof, ' '.join([str(val) for val in unitcell]), pressure)

    f.write(dedent(simulation_file_header))

    component_number = 0
    for gas in  composition.keys():
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

                 """ % (component_number, gas, mole_fraction)

        f.write(dedent(simulation_file_gas))
        component_number += 1

    f.close()


def write_all_raspa_files_and_database(config_file):
    data = yaml_loader(config_file)
    mofs, unitcells = read_mof_configuration_csv(data['mofs_filepath'])
    compositions = read_composition_configuration(data['comps_filepath'])
    pressure = data['pressure']

    all_input_files = []

    # 1. Make main directory with timestamp of submission
    main_dir = generate_unique_run_name()
    os.makedirs(main_dir)

    for i in range(len(mofs)):
        mof, unitcell = mofs[i], unitcells[i]

        # 2. Generate unique directory per MOF
        mof_dir = os.path.join(main_dir, mof)
        os.makedirs(mof_dir)

        # 3. Setup .csv file and write header
        f = open(os.path.join(mof_dir, mof+'.csv'),'w',newline='')
        header = ['MOF']
        header.extend([gas+'_comp' for gas in compositions.keys()])
        for gas in compositions.keys():
            header.extend([gas+'_mass', gas+'_error'])
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)

        # Queue simulations
        for comp_id, composition in compositions.iterrows():
            # 4. Create unique working directory for each simulation
            sim_dir = os.path.join(mof_dir, str(comp_id))
            os.makedirs(sim_dir, exist_ok=True)

            # 5. Write the input file and run the simulation.
            raspa_input_file = os.path.join(sim_dir, "simulation.input")
            write_raspa_file(raspa_input_file, mof, unitcell, composition, pressure)
            subprocess.run(['sbatch', '../launch_workers.slurm'], check=True, cwd=sim_dir)
