'''
Notebook to Generate Gas Mixtures
  Original Code by: Jenna Gustafson
  Modifications by: Brian Day

Required Inputs:
  - Number of components
  - Lower and upper composition limits for 2 of 3 components
  - Difference between mole fractions

Notes:
  - Currently only works for a maximum of 3 components
'''

import numpy as np
import itertools
import csv

# Specify the gases
gases = ['CO2','O2','N2']
num_components = len(gases)

# Specify the ranges for 2 components
limits0 = [0.00 , 0.30]
limits1 = [0.00 , 0.30]
LL2 = 1 - (limits0[1] + limits1[1])
if LL2 < 0:
    LL2 = 0
UL2 = 1 - (limits0[0] + limits1[0])
if UL2 > 1:
    UL2 = 1
limits2 = [LL2 , UL2]

# Specify the change in composition
divisions = 0.01
num_vals0 = ((limits0[1]-limits0[0])/divisions) + 1
numbers0 = np.linspace(limits0[0], limits0[1], int(num_vals0))
num_vals1 = ((limits1[1]-limits1[0])/divisions) + 1
numbers1 = np.linspace(limits1[0], limits1[1], int(num_vals1))
num_vals2 = ((limits2[1]-limits2[0])/divisions) + 1
numbers2 = np.linspace(limits2[0], limits2[1], int(num_vals2))

np.set_printoptions(formatter={'float': '{: 1.2f}'.format})
print('Points for', gases[0], ' = \n', numbers0)
print('\nPoints for', gases[1], ' = \n', numbers1)
print('\nPoints for', gases[2], ' = \n', numbers2)

# Generate array of all possible mixtures
k = 0
comp_list = np.array([])
for i in range(0,int(num_vals0)):
    for j in range(0,int(num_vals1)):
        comp0 = numbers0[i]
        comp1 = numbers1[j]
        comp2 = np.round(1-numbers0[i]-numbers1[j], 4)
        if comp2 >= 0 and comp2 <= 1:
            new_entry = np.array([numbers0[i], numbers1[j], np.round(1-numbers0[i]-numbers1[j], 4)])
            comp_list = np.concatenate((comp_list, new_entry), axis=0)
            k = k+1
print('\nTotal number of mixtures = ', k)
comp_list = np.reshape(comp_list, (k,3))
print('\nList of Compositions: \n', comp_list)
