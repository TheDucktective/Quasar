from qsofunc import *
warnings.filterwarnings("ignore")
from scipy.signal import medfilt
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import glob, os, sys, timeit
import numpy as np
sys.path.append('../')
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from quasar_fetch import Fetch
import numpy
from scipy.interpolate import make_smoothing_spline
import pandas as pd


def smooth(y, window):
    box = np.ones(window)/window
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


# Définition des ID du spectre
plate,mjd,fiberID,z =7832,56904,730,2.0598

# Téléchargement du fichier du FIT du spectre
spec_url =f'http://quasar.astro.illinois.edu/paper_data/\
DR16Q/fits/{plate:04d}/op-{plate:04d}-{mjd}-{fiberID:04d}.fits.gz'

# Attribution des différentes données du fit
op_data2 = fits.getdata(spec_url ,1)
op_data2_MC = fits.getdata(spec_url ,2)
op_data2_spec = fits.getdata(spec_url ,3)

print("Download successful")

# Setting plot title with quasar ID
plot_title = 'op-'+"%04d" %(    plate)+'-'+"%05d" %(mjd)+'-'+"%04d" %(fiberID)+'    z ='\
    +str(z)

# read continuum fitting parameters
conti_keyword = ['PL_norm', 'PL_slope',  'POLY_a', 'POLY_b', 'POLY_c',\
                 'Fe_uv_norm', 'Fe_uv_FWHM', 'Fe_uv_shift', 'Fe_op_norm', 'Fe_op_FWHM', 'Fe_op_shift']
conti_para = np.array([op_data2[name][0] for name in conti_keyword])
conti_model = continuum_all(op_data2_spec.wave_prereduced, conti_para)

# power law + poly continuum model
pl_model = conti_para[0]*(op_data2_spec.wave_prereduced/3000.0)**conti_para[1]
poly_model = F_poly_conti(op_data2_spec.wave_prereduced, conti_para[2:5])
pl_poly_model = pl_model + poly_model

# complex line lists
op_linelist = np.array(['HALPHA', 'HALPHA_BR', 'NII6585', 'SII6718', \
                        'HBETA', 'HBETA_BR', 'HEII4687', 'HEII4687_BR', \
                        'OIII5007', 'OIII5007C', 'CAII3934', 'OII3728', 'NEV3426', \
                        'MGII', 'MGII_BR', 'CIII_ALL', 'CIII_BR', \
                        'SIIII1892', 'ALIII1857', 'NIII1750', \
                        'CIV', 'HEII1640', 'HEII1640_BR', \
                        'SIIV_OIV', 'OI1304', 'LYA', 'NV1240'])

# fitted complex line lists
comp_lst = np.array([op_data2[hdn][0] for hdn in op_data2.names if ('complex_name' in hdn) and ('local' not in hdn)])
ncomp = len(comp_lst)

# fitted individual line lists
line_lst = np.array([hdn[:-6] for hdn in op_data2.names if ('_scale' in hdn) and ('err' not in hdn)])

# calculate the fitted gaussian line fit models
line_flux = np.zeros(len(op_data2_spec.wave_prereduced))
for l in range(len(line_lst)):
    line_gauss = np.array([op_data2[line_lst[l]+'_scale'], op_data2[line_lst[l]+'_centerwave'], \
                           op_data2[line_lst[l]+'_sigma']])
    line_flux += Manygauss(np.log(op_data2_spec.wave_prereduced), line_gauss)

# major lines wavelength
line_cen = np.array([6564.61, 6732.66, 4862.68, 5008.24, 4687.02, 4341.68, 3934.78, 3728.47, \
                     3426.84, 2798.75, 1908.72, 1816.97, 1750.26, 1718.55, 1549.06, 1640.42, 1402.06, 1396.76, 1335.30, \
                     1304, 1215.67])
# major lines names
line_name = np.array(['Ha+[NII]', '[SII]6718,6732', 'Hb', '[OIII]', 'HeII4687', 'Hr', 'CaII3934', '[OII]3728', \
                      'NeV3426', 'MgII', 'CIII]', 'SiII1816', 'NIII]1750', 'NIV]1718', 'CIV', 'HeII1640', '',
                      'SiIV+OIV', \
                      'CII1335','OI1304','Lya'])

# processed spectrum
fig, ax = plt.subplots(1, 1, figsize=(15, 4))

# commonly seen quasar lines
med_flux = medfilt(op_data2_spec.flux_prereduced, kernel_size=5)
for ll in range(len(line_cen)):
    if op_data2_spec.wave_prereduced.min() < line_cen[ll] < op_data2_spec.wave_prereduced.max():
        ax.axvline(line_cen[ll], c='grey', ls='--', zorder=1)
        ax.text(line_cen[ll] + 7, med_flux.max() * 1.15, line_name[ll], rotation=90, fontsize=10, va='top', zorder=5)

# plot QSOFit model
fig, ax = plt.subplots(2, 1, figsize=(15, 4))
#ax.tick_params(which="both",bottom=True, top=True, left=True, right=True)
ax[0].errorbar(op_data2_spec.wave_prereduced, op_data2_spec.flux_prereduced, yerr=op_data2_spec.err_prereduced, \
            color='k', ecolor='silver', label='data', zorder=1)
ax[0].plot(op_data2_spec.wave_prereduced, line_flux, c='seagreen', label='line fit', zorder=2)
ax[0].plot(op_data2_spec.wave_prereduced, conti_model+line_flux, c='red', label='Fit', zorder=2)
ax[0].plot(op_data2_spec.wave_prereduced, conti_model, c='orange', label='continuum fit', zorder=5)
#ax[0].plot(op_data2_spec.wave_prereduced[ind_abs], op_data2_spec.flux_prereduced[ind_abs], \
#        ls='', marker='x', color='cornflowerblue', ms=3, zorder=7)
ax[1].plot(op_data2_spec.wave_prereduced, op_data2_spec.flux_prereduced/(conti_model+line_flux), c='k', label='Normalized flux')
ax[1].set_xlim(1205,2700)
ax[1].set_ylim(-2.5,5)
ax[0].legend(loc='upper right', ncol=2, edgecolor='None', facecolor='w', fontsize=14)
ax[0].axhline(0, ls='--', c='grey', zorder=0)
ax[0].xaxis.set_major_locator(MultipleLocator(250.))
ax[0].xaxis.set_minor_locator(MultipleLocator(50.))
ax[0].yaxis.set_major_locator(MultipleLocator(100.))
ax[0].yaxis.set_minor_locator(MultipleLocator(50.))
ax[0].set_title(plot_title)
ax[0].set_xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)')
ax[0].set_ylabel(r'$ f_{\lambda}$ ($\rm 10^{-17} {\rm erg\;s^{-1}\;cm^{-2}\;\AA^{-1}}$)')
ax[0].set_xlim(op_data2_spec.wave_prereduced.min(), op_data2_spec.wave_prereduced.max())
med_flux = medfilt(op_data2_spec.flux_prereduced, kernel_size=5)
ax[0].set_ylim(-1.2*abs(med_flux.min()), 1.2*med_flux.max())
set_mpl_style(major=8.0,minor=4.0,lwidth=1.2)

# major lines
line_cen = np.array([6564.61, 6732.66, 4862.68, 5008.24, 4687.02, 4341.68, 3934.78, 3728.47, \
                     3426.84, 2798.75, 1908.72, 1816.97, 1750.26, 1718.55, 1549.06, 1640.42, 1402.06, 1396.76, 1335.30, \
                     1215.67])
line_name = np.array(['Ha+[NII]', '[SII]6718,6732', 'Hb', '[OIII]', 'HeII4687', 'Hr', 'CaII3934', '[OII]3728', \
                      'NeV3426', 'MgII', 'CIII]', 'SiII1816', 'NIII]1750', 'NIV]1718', 'CIV', 'HeII1640', '',
                      'SiIV+OIV', \
                      'CII1335', 'Lya'])

# complex line list
complex_lines = np.array(['Ha', 'Hb', 'MgII', \
                          'CIII', 'CIV', 'SiIV', 'Lya', \
                          ])

# complex line range for plot
complex_line_range = np.array([[6400, 6800], [4640, 5100], [2700, 2900], \
                               [1700, 1970], [1500, 1700], [1290, 1450], [1150, 1290], \
                               ])

# adjust the plotting parameter for each lines
subline_xloc = np.array([70, 80, 100, 95, 100])
subline_yloc = np.array([50, 20, 20, 10, 10])
subline_yloc_range = np.array([[-50, 160], [-20, 42], [-19, 47], [-7, 32], [-6, 19]])

fig, axes = plt.subplots(1, 1, figsize=(15, 8))
set_mpl_style(major=8.0, minor=4.0, lwidth=1.2)
ax1 = axes
ax1.set_position([0.07, 0.54, 0.88, 0.40])

# plot the top panel -- overall view of the qsofit result
ax1.errorbar(op_data2_spec.wave_prereduced*(1+z), op_data2_spec.flux_prereduced, yerr=op_data2_spec.err_prereduced, \
             color='k', ecolor='silver', label='data', zorder=1)

med_flux = medfilt(op_data2_spec.flux_prereduced, kernel_size=5)
ax1.plot(op_data2_spec.wave_conti, 1.2 * med_flux.max() * np.ones(len(op_data2_spec.wave_conti)), \
         color='grey', ls='', marker='s', zorder=1)  # the continuum fitting windows
ax1.set_xlim(op_data2_spec.wave_prereduced.min()*(1+z), op_data2_spec.wave_prereduced.max()*(1+z))
ax1.set_ylim(-1.2 * abs(med_flux.min()), 1.2 * med_flux.max())

# commonly seen quasar lines
for ll in range(len(line_cen)):
    if op_data2_spec.wave_prereduced.min() < line_cen[ll] < op_data2_spec.wave_prereduced.max():
        ax1.axvline(line_cen[ll]*(1+z), c='grey', ls='--', zorder=1)
        ax1.text(line_cen[ll]*(1+z) + 7, med_flux.max() * 1.15, line_name[ll], rotation=90, fontsize=10, va='top', zorder=5)

# ax1.legend(loc='upper right', ncol=2, edgecolor='None', facecolor='w', fontsize=14)
ax1.axhline(0, c='grey', ls='--', zorder=1)
ax1.set_title(plot_title)
ax1.xaxis.set_ticks_position('both')
ax1.yaxis.set_ticks_position('both')
ax1.xaxis.set_major_locator(MultipleLocator(500.))
ax1.xaxis.set_minor_locator(MultipleLocator(100.))
ax1.yaxis.set_major_locator(MultipleLocator(50.))
ax1.yaxis.set_minor_locator(MultipleLocator(10.))
ax1.tick_params(axis='x', which='both', direction='in')
ax1.tick_params(axis='y', which='both', direction='in')

# plot the bottom panel -- zoom-in view of complex lines
fig_dx = 0.03
fig_x = (0.88 - (ncomp - 1) * fig_dx) / ncomp
s = 0
for n in range(ncomp):
    ind_comp = np.where(complex_lines == comp_lst[n])[0]
    ind_line = np.where((op_data2_spec.wave_prereduced > complex_line_range[ind_comp, 0]) \
                        & (op_data2_spec.wave_prereduced < complex_line_range[ind_comp, 1]))[0]
    ind_line2 = np.where((op_data2_spec.wave_line_abs > complex_line_range[ind_comp, 0]) \
                         & (op_data2_spec.wave_line_abs < complex_line_range[ind_comp, 1]))[0]

    axn = fig.add_subplot(1, ncomp, n + 1)
    axn.set_position([0.07 + n * fig_x + n * fig_dx, 0.08, fig_x, 0.40])
    axn.errorbar(op_data2_spec.wave_prereduced[ind_line]*(1+z), op_data2_spec.flux_prereduced[ind_line] - conti_model[ind_line], \
                 yerr=op_data2_spec.err_prereduced[ind_line], \
                 color='k', ecolor='silver', zorder=1)
    axn.errorbar(op_data2_spec.wave_prereduced[ind_line]*(1+z), line_flux[ind_line], \
                 color='red', zorder=4)
    #axn.plot(op_data2_spec.wave_line_abs[ind_line2], op_data2_spec.flux_line_abs[ind_line2], \
    #         ls='', marker='x', color='cornflowerblue', ms=5, zorder=7)

    for ll in line_lst:
        if complex_line_range[ind_comp, 0] < np.exp(op_data2[ll + '_centerwave']) < complex_line_range[ind_comp, 1]:
            line_gauss_para = np.array(
                [op_data2[ll + '_scale'][0], op_data2[ll + '_centerwave'][0], op_data2[ll + '_sigma'][0]])
            if 2 * np.sqrt(2 * np.log(2)) * (np.exp(op_data2[ll + '_sigma'][0]) - 1) * 3.e5 < 1200.:
                line_color, line_order = 'b', 2
            else:
                line_color, line_order = 'seagreen', 3
            axn.plot(op_data2_spec.wave_prereduced[ind_line], \
                     Manygauss(np.log(op_data2_spec.wave_prereduced[ind_line]), line_gauss_para), \
                     color=line_color, zorder=line_order)

    axn.text(0.62, 0.85, r'$\chi ^2_r=$' + str(np.round(op_data2[str(n + 1) + '_line_red_chi2'][0], 2)),
             fontsize=12, transform=axn.transAxes)
    if comp_lst[n] == 'CIII':
        axn.text(0.79, 0.92, comp_lst[n] + ']', fontsize=14, transform=axn.transAxes)
    else:
        axn.text(0.79, 0.92, comp_lst[n], fontsize=14, transform=axn.transAxes)
    axn.axhline(0, c='grey', ls='--', zorder=1)
    axn.set_xlim(complex_line_range[ind_comp, 0], complex_line_range[ind_comp, 1])
    axn.set_ylim(subline_yloc_range[s, 0], subline_yloc_range[s, 1])
    axn.xaxis.set_ticks_position('both')
    axn.yaxis.set_ticks_position('both')
    axn.xaxis.set_major_locator(MultipleLocator(subline_xloc[s]))
    axn.xaxis.set_minor_locator(MultipleLocator(subline_xloc[s] / 5.))
    axn.yaxis.set_major_locator(MultipleLocator(subline_yloc[s]))
    axn.yaxis.set_minor_locator(MultipleLocator(subline_yloc[s] / 5.))
    s += 1
    # axn.tick_params(axis='x',which='both',direction='in')
    # axn.tick_params(axis='y',which='both',direction='in')

ax1.text(0.5, -1.31, r'$\rm Observed \, Wavelength$ ($\rm \AA$)', fontsize=18, transform=ax1.transAxes,
         ha='center')
ax1.text(-0.05, -0.05, r'$ f_{\lambda}$ ($\rm 10^{-17} {\rm erg\;s^{-1}\;cm^{-2}\;\AA^{-1}}$)', fontsize=18,
         transform=ax1.transAxes, rotation=90, ha='center', rotation_mode='anchor')

plt.show()


'''
SMOOTHING OF THE SPECTRUM
'''
def remove_outliers(spikey, baseline, cutoff):
    ''' Remove data from df_spikey that is > delta from fbewma. '''
    np_spikey = np.array(spikey)
    np_baseline = np.array(baseline)
    cond_delta = (abs(np_spikey-np_baseline) > cutoff)
    np_remove_outliers = np.where(cond_delta, np.nan, np_spikey)
    return np_remove_outliers


baseline = smooth(op_data2_spec.flux_prereduced,31)
remove_spike = remove_outliers(op_data2_spec.flux_prereduced,baseline, 30)
interpolate = pd.DataFrame(remove_spike).interpolate()
smooth = smooth(interpolate.to_numpy().ravel(),31)


# processed spectrum
fig, ax = plt.subplots(1, 1, figsize=(15, 4))
ax.tick_params(which="both",bottom=True, top=True, left=True, right=True)
ax.errorbar(op_data2_spec.wave_prereduced, op_data2_spec.flux_prereduced, yerr=op_data2_spec.err_prereduced, \
            color='k', ecolor='silver', label='data', zorder=1)
ax.plot(op_data2_spec.wave_prereduced, baseline, color="red")
ax.xaxis.set_major_locator(MultipleLocator(250.))
ax.xaxis.set_minor_locator(MultipleLocator(50.))
ax.yaxis.set_major_locator(MultipleLocator(100.))
ax.yaxis.set_minor_locator(MultipleLocator(50.))
ax.set_title(plot_title)
ax.set_xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)')
ax.set_ylabel(r'$ f_{\lambda}$ ($\rm 10^{-17} {\rm erg\;s^{-1}\;cm^{-2}\;\AA^{-1}}$)')
ax.set_xlim(op_data2_spec.wave_prereduced.min(), op_data2_spec.wave_prereduced.max())
med_flux = medfilt(op_data2_spec.flux_prereduced, kernel_size=5)
ax.set_ylim(-1.2*abs(med_flux.min()), 1.2*med_flux.max())
set_mpl_style(major=8.0,minor=4.0,lwidth=1.2)
plt.show()

# processed spectrum
fig, ax = plt.subplots(1, 1, figsize=(15, 4))
ax.tick_params(which="both",bottom=True, top=True, left=True, right=True)
#ax.errorbar(op_data2_spec.wave_prereduced, op_data2_spec.flux_prereduced, yerr=op_data2_spec.err_prereduced, \
#            color='k', ecolor='silver', label='data', zorder=1)
ax.plot(op_data2_spec.wave_prereduced, op_data2_spec.flux_prereduced/smooth, color="black", label="smooth")
ax.plot(op_data2_spec.wave_prereduced, op_data2_spec.flux_prereduced/(conti_model+line_flux), color="red", label="PQSOFit")
ax.axhline(1, linestyle='--', color='orange')
ax.xaxis.set_major_locator(MultipleLocator(250.))
ax.xaxis.set_minor_locator(MultipleLocator(50.))
ax.yaxis.set_major_locator(MultipleLocator(100.))
ax.yaxis.set_minor_locator(MultipleLocator(50.))
ax.set_title(plot_title)
ax.set_xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)')
ax.set_ylabel('Normalized flux')
ax.set_xlim(op_data2_spec.wave_prereduced.min(), op_data2_spec.wave_prereduced.max())
med_flux = medfilt(op_data2_spec.flux_prereduced, kernel_size=5)
plt.xticks(np.arange(min(op_data2_spec.wave_prereduced)-2, max(op_data2_spec.wave_prereduced)-2, 50.0))
plt.yticks(np.arange(-2, 5, 0.25))
ax.legend(loc='upper right', ncol=2, edgecolor='None', facecolor='w', fontsize=12)
set_mpl_style(major=8.0,minor=4.0,lwidth=1.2)
plt.show()
