import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def plot_adsorbed_masses(ads_data, gas, figname=None):

    comps = ads_data[gas+'_comp']
    mass = ads_data[gas+'_mass']
    error = ads_data[gas+'_error']
    air_mass = ads_data['O2_mass']+ads_data['N2_mass']
    air_error = ads_data['O2_error']+ads_data['N2_error']
    total_mass = ads_data['total_mass']
    total_error = ads_data['total_mass_error']

    plt.clf()
    plt.figure(figsize=(4.5,4.5), dpi=100)
    plt.errorbar(comps, mass, error, marker='o', markersize=5, alpha=0.5, linewidth=0, elinewidth=1)
    plt.errorbar(comps, air_mass, yerr=air_error, marker='o', markersize=5, alpha=0.5, linewidth=0, elinewidth=1)
    plt.errorbar(comps, total_mass, yerr=total_error, marker='o', markersize=5, alpha=0.1, linewidth=0, elinewidth=1)
    plt.xlabel(gas+' Mole Fractions')
    plt.ylabel('Adsorbed Mass')
    plt.legend([gas, 'Air', 'Total'])
    plt.tight_layout()

    if figname != None:
        plt.savefig(figname)
    plt.close()


def plot_kH(ads_data, gas, p, figname=None):

    comps = ads_data[gas+'_comp']
    mass = ads_data[gas+'_mass']
    error = ads_data[gas+'_error']
    air_mass = ads_data['O2_mass']+ads_data['N2_mass']
    air_error = ads_data['O2_error']+ads_data['N2_error']
    total_mass = ads_data['total_mass']
    total_error = ads_data['total_mass_error']

    # Initialize Figure
    plt.clf()
    plt.figure(figsize=(4.5, 4.5), dpi=100)

    fit = np.poly1d(p)
    plt.plot(comps, fit(comps), 'r-')
    plt.errorbar(comps, mass, error, marker='o', markersize=3, elinewidth=1, linewidth=0)

    plt.xlabel(gas+' Mole Fractions')
    plt.xticks(np.linspace(0,0.05,6), fontsize=8)
    plt.ylabel('Adsorbed Mass\n[mg/g Framework]')
    plt.yticks(fontsize=8)

    textstr='K_H = '+str(np.round(p[0], 2))
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    plt.text(0.3*max(comps), 1.08*max(mass)+max(error), textstr, fontsize=10, bbox=props)

    plt.tight_layout()
    if figname != None:
        plt.savefig(figname)
    plt.close()


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
