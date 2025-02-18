# MOF Sensor Array: GCMC Simulations & Array Analysis

[![eNose Test Status](https://github.com/birdday/enose/actions/workflows/python-app.yml/badge.svg?branch=master-birdday)](https://github.com/birdday/enose/actions/workflows/python-app.yml)

# Installation
Clone the repository located in the [WilmerLab GitHub page] (https://github.com/WilmerLab/sensor_array_mof_adsorption).

### Local Installation
* Install the requirements file, requirements_local.txt, located in the **requirements** folder with:
```
pip install -r requirements/requirements_local.txt
```

### Installation for SMP Cluster on H2P (University of Pittsburgh)
1. Set up your virtual environment.
      * Follow the instructions for setting up a virtual environment via the README.md in <https://github.com/paulboone/sam-subjobserver>.
      * Step 2 requires configuring the correct version of python (see README link above)  
2. Install the requirements in the file requirements_h2p.txt, located  in the **requirements** folder via pip install.
3. Copy file launch_workers.sample.slurm to launch_workers.slurm and make any necessary changes.
      * Change any configurations for h2p
4. Copy file sjs.sample.yaml located in **settings** directory to sjs.yaml, changing database number.
      * Please consult Paul Boone @ paulboone@pitt.edu for choosing a database number (1-12).
      * Changing the queue name is not necessary   
5. Copy file rq_worker_config.sample.py located in **settings** directory to rq_worker_config.py.
      * Change the number at the end of the redis URL from 0 to the database you chose from step 4
      * Change the queue name to the name of the queue specified in step 4

### Test the Installation
* For both local and cluster installations, use the files provided in the **example** directory to
test the installation. Files are provided in the **example** directory for testing the code. See the
usage section below for how to submit these files.

N.B. For all simulations, the gas mixture compositions are imported in terms of mole fractions. Also,
make sure that your comps.csv file is tab delimited or else it will result in an error.

N.B. You may need to install the dependencies for matplotlib with:
```
sudo apt-get install libpng-dev libfreetype6-dev
```

<br/><br/>

#  GCMC (RASPA) Simulations
### How to Run the Simulations: Local
* Execute the script to run the simulations in serial mode, requiring input arguments of the mofs list,
compositions list, gas list, and pressure (Pa). All simulations use room temperature. Below is an example at atmospheric pressure.
```
./write_gcmc_simulation.py example/mofs_test.csv example/comps_test.csv example/gas_list.csv 1E5
```
* The results will save as a table format in a file comps_mass_output.csv, contaning a run ID, the MOF,
gas mixture composition, and total mass adsorbed (mg gas/cm3 framework).
* Located in the **settings** directory, copy write_gcmc_sim_config.sample.yaml to your own file
write_gcmc_sim_config.yaml and make any necessary changes. For each gas in the gas mixtures you are
simulating, list the name you are importing the gas as (eg. CO2, CH4) and the name of its
corresponding .def file (eg. CO2, methane) in raspa.

### How to Run the Simulations: Remote
* See above for executing simulation script, this will be the same for local and remote cases. Here,
there will be a note saying the jobs were queued on the server. Once they are queued, you must launch
the workers on the HPC cluster by executing the launch_workers_qsub.sh script. This will start the job server.
```
sbatch launch_workers.slurm
```
* You may view the status of the workers by typing into the command prompt:
```
rq info -u redis://10.201.0.11:6379/0
```
* Where 0 is the database on which you have been given to run your jobs. For more information regarding the
job server, please refer to the README on the https://github.com/paulboone/sam-subjobserver.

<br/><br/>

# Array Analysis
### General Information
* Located in the **settings** directory, copy process_config.sample.yaml to your own file
process_config.yaml and make any necessary changes. You will need to provide the densities of the
MOFs which you are simulating, and the **experimental** mass, or the total mass adsorbed by each MOF
for one specific gas mixture of your choosing. (This value may be an approximation from any known
simulation data or may be from experimental results.)
### Brute Force (i.e. All Possible Arrays)
### Genetic Algorithm

<br/><br/>

# License
