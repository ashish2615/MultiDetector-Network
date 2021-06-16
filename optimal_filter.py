#!/usr/bin/env python3
from __future__ import division, print_function

import os
import sys

import bilby
from bilby.core.utils import speed_of_light

import deepdish
import numpy as np
import logging
import deepdish
import pandas as pd
import json
import math
import sklearn
import seaborn as sns

import scipy
from scipy import signal, fftpack
from scipy.fftpack import fft, rfft, ifft, irfft , fftfreq, rfftfreq
from scipy.signal import *

import astropy
from astropy import constants as const
from astropy.constants import G, pc, c

import overlap_reduction_function
from overlap_reduction_function import ifos, IFOs, sampling_frequency, duration_seg, frequency, odf

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

current_direc = os.getcwd()
print("Current Working Directory is :", current_direc)

## Specify the output directory and the name of the simulation.
outdir = 'Optimal_Filter'
label = 'Optimal_filter_'
bilby.utils.setup_logger(outdir=outdir, label=label)

#################################
''' Optimal Signal Processing '''
#################################

gamma = odf(frequency)
# print('gamma',gamma)
# print(np.shape(gamma))

# SGWB and noise intrinsic to detector are stationary.
# Cross-correlation signal between the output of two detectors

## create a time series
time_series = bilby.core.utils.create_time_series(sampling_frequency, duration_seg, starting_time=0.)
## Create a frequency Series
frequency_series =  bilby.core.utils.create_frequency_series(sampling_frequency, duration_seg)

## Set One sided Noise Power Spectral Density of the detectors (because frequecy is from 0 to higher values).
psd = np.zeros((len(IFOs),len(frequency)))
ci = 0
for ifo in IFOs:
    PSD = ifo.power_spectral_density
    psd[ci,] = PSD.power_spectral_density_interpolated(frequency)
    ci += 1

psd[np.isinf(psd)] = 0
psd[np.isnan(psd)] = 0

print('psd is',(psd))

## Spectrum of Stochastic Gravitaional wave background
## Hubble constant H0 = (67.4±0.5) km s−1Mpc−1
G = 6.67408 * 10 ** -11                        ## units = m**3/ (kg* sec**2)
one_pc = 3.0856775814671916 * 10 ** 16         ## units = m
H0 = 67.4 * 10 ** 3 * 10 ** -6 * one_pc ** -1  ## units = 1/sec
# print(H0)

# h0 = 0.6766
# H0 = h0*3.24*10**-18
# print(H0)

## rho_c = (3 * c ** 2 * H0 ** 2)/(8 * np.pi * G)
rho_c = (3 * speed_of_light ** 2 * H0 ** 2)/(8 * np.pi * G)   ## units = erg/cm**3
omega_gw = 10 ** -15

# ## Normalization Constant : PHYSICAL REVIEW D 69, 122004 ~2004!
# norm_c = (20 * np.pi ** 2) / (3 * H0 ** 2) * (
#             ((gamma(frequency) ** 2) / (frequency ** 3 * psd[0, :] * psd[1, :])) * (frequency[1] - frequency[0])) ** -1
# norm_c[numpy.isinf(norm_c)] = 0
# norm_c[numpy.isnan(norm_c)] = 0

class optimal_filter_calculation:

    ## Optimal filter depends upon the location and orientation of detector as well as SGWB and Noise PSD of detector.
    ## Allen and Romano 1999 Eq. 3.73
    def optimal_filter_AR(self,frequency):

        optimal_filter_AR = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), len(frequency)))
        cidx = 0
        for idx in np.arange((int(len(IFOs) * (len(IFOs) - 1) / 2))):

            optimal_filter_AR[cidx, :] = (gamma[idx,:] * omega_gw) / (frequency ** 3 * psd[0, :] * psd[1, :])
            cidx += 1

        return  optimal_filter_AR

    ## Jan Paper
    def optimal_filter_JH(self,frequency):

        ## Background spectral density S^b(frequency)
        optimal_filter_JH = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), len(frequency)))
        cidx = 0
        for idx in np.arange((int(len(IFOs) * (len(IFOs) - 1) / 2))):
            Sb = np.dot((3. * H0 ** 2) / (4 * np.pi ** 2), np.divide(omega_gw, frequency ** 3))
            Sb[np.isinf(Sb)] = 0

            optimal_filter_JH[cidx, :] = (gamma[idx, :] * Sb) / (psd[0, :] * psd[1, :])
            cidx += 1

        return optimal_filter_JH

optimal_filter = optimal_filter_calculation()

of_AR = optimal_filter.optimal_filter_AR(frequency)
of_AR[np.isinf(of_AR)] = 0
of_AR[np.isnan(of_AR)] = 0
print('of_AR',of_AR)
print(np.shape(of_AR))

# for idx in np.arange(15):
#
#     plt.figure(idx)
#
#     if idx == 0:
#         label = label = 'LIGO Livingston AND Hanford'
#     if idx == 1:
#         label = label = 'LIGO Livingston AND CE'
#     if idx == 2:
#         label = label = 'LIGO Livingston AND ET_D_TR_1'
#     if idx == 3:
#         label = label = 'LIGO Livingston AND ET_D_TR_2'
#     if idx == 4:
#         label = label = 'LIGO Livingston AND ET_D_TR_3'
#     if idx == 5:
#         label = label = 'LIGO Hanford AND CE'
#     if idx == 6:
#         label = label = 'LIGO Hanford AND ET_D_TR_1'
#     if idx == 7:
#         label = label = 'LIGO Hanford AND ET_D_TR_2'
#     if idx == 8:
#         label = label = 'LIGO Hanford  AND ET_D_TR_3'
#     if idx == 9:
#         label = label = 'CE AND ET_D_TR_1'
#     if idx == 10:
#         label = label = 'CE AND ET_D_TR_2'
#     if idx == 11:
#         label = label = 'CE AND ET_D_TR_3'
#     if idx == 12:
#         label = label = 'ET_D_TR_1  AND ET_D_TR_2'
#     if idx == 13:
#         label = label = 'ET_D_TR_1  AND ET_D_TR_3'
#     if idx == 14:
#         label = label = 'ET_D_TR_2  AND ET_D_TR_3'
#
#     plt.plot(frequency, of_AR[idx, :], label = label)
#     legend = plt.legend(loc='lower right', fontsize='small')
#     # plt.xscale('log')
#     # plt.xlim(1, 1000)
#     plt.xlim(0, 600)
#     plt.xlabel(r'f (Hz)')
#     plt.ylabel(r'of_AR(f)')
#     plt.title(r' Optimal_filter Between ' + label)
#     plt.savefig('./Optimal_filter/Optimal_filter_AR_ ' + label)
#     # plt.savefig('./Optimal_filter/Optimal_filter_AR__log_ ' + label)
#     plt.show()
#     plt.close()

plt.figure(16)
plt.plot(frequency, of_AR[0,:], label = 'CE & ET_D_TR_1' )
plt.plot(frequency, of_AR[1,:], label = 'CE & ET_D_TR_2' )
plt.plot(frequency, of_AR[2,:], label = 'CE & ET_D_TR_3' )
legend = plt.legend(loc='lower right', fontsize='small')
# plt.xscale('log')
# plt.xlim(1, 1000)
plt.xlim(0, 600)
# plt.axhline(y=0)
# plt.axvline(x=0)
plt.xlabel(r'f (Hz)')
plt.ylabel(r'of_AR(f)')
plt.title(r' Optimal_filter Between CE and ET_D_TR' )
plt.savefig('./Optimal_Filter/Optimal_filter_AR Between CE and ET_D_TR')
# plt.savefig('./Optimal_Filter/Optimal_filter_AR__log Between CE and ET_D_TR ')
plt.show()
plt.close()

of_JH = optimal_filter.optimal_filter_JH(frequency)
of_JH[np.isinf(of_JH)] = 0
of_JH[np.isnan(of_JH)] = 0
print('of_JH',of_JH)
print(np.shape(of_JH))

# for idx in np.arange(15):
#
#     plt.figure(idx)
#
#     if idx == 0:
#         label = label = 'LIGO Livingston AND Hanford'
#     if idx == 1:
#         label = label = 'LIGO Livingston AND CE'
#     if idx == 2:
#         label = label = 'LIGO Livingston AND ET_D_TR_1'
#     if idx == 3:
#         label = label = 'LIGO Livingston AND ET_D_TR_2'
#     if idx == 4:
#         label = label = 'LIGO Livingston AND ET_D_TR_3'
#     if idx == 5:
#         label = label = 'LIGO Hanford AND CE'
#     if idx == 6:
#         label = label = 'LIGO Hanford AND ET_D_TR_1'
#     if idx == 7:
#         label = label = 'LIGO Hanford AND ET_D_TR_2'
#     if idx == 8:
#         label = label = 'LIGO Hanford  AND ET_D_TR_3'
#     if idx == 9:
#         label = label = 'CE AND ET_D_TR_1'
#     if idx == 10:
#         label = label = 'CE AND ET_D_TR_2'
#     if idx == 11:
#         label = label = 'CE AND ET_D_TR_3'
#     if idx == 12:
#         label = label = 'ET_D_TR_1  AND ET_D_TR_2'
#     if idx == 13:
#         label = label = 'ET_D_TR_1  AND ET_D_TR_3'
#     if idx == 14:
#         label = label = 'ET_D_TR_2  AND ET_D_TR_3'
#
#     plt.plot(frequency, of_JH[idx, :], label = label)
#     legend = plt.legend(loc='lower right', fontsize='small')
#     # plt.xscale('log')
#     # plt.xlim(1, 1000)
#     plt.xlim(0, 600)
#     plt.xlabel(r'f (Hz)')
#     plt.ylabel(r'of_JH (f)')
#     plt.title(r' Optimal_filter Between' + label)
#     plt.savefig('./Optimal_filter/Optimal_filter_JH_' + label)
#     # plt.savefig('./Optimal_filter/Optimal_filter_JH_log_ ' + label)
#     plt.show()
#     plt.close()

plt.figure(34)
plt.plot(frequency, of_JH[0,:], label = 'CE & ET_D_TR_1' )
plt.plot(frequency, of_JH[1,:], label = 'CE & ET_D_TR_2' )
plt.plot(frequency, of_JH[2,:], label = 'CE & ET_D_TR_3' )
legend = plt.legend(loc='lower right', fontsize='small')
# plt.xscale('log')
# plt.xlim(1, 1000)
plt.xlim(0, 600)
# plt.axhline(y=0)
# plt.axvline(x=0)
plt.xlabel(r'f (Hz)')
plt.ylabel(r'of_AR(f)')
plt.title(r' Optimal_filter JH Between CE and ET_D_TR' )
plt.savefig('./Optimal_Filter/Optimal_filter_JH Between CE and ET_D_TR')
# plt.savefig('./Optimal_Filter/Optimal_filter_JH__log Between CE and ET_D_TR ')
plt.show()
plt.close()


## Background spectral density S^b(frequency)
## Jan Paper
Sb = np.dot((3. * H0 ** 2) / (4 * np.pi ** 2), np.divide(omega_gw, frequency ** 3))
Sb[np.isinf(Sb)] = 0

## mean value of cross-correlated signal is. A factor two in mu is beacuse limit has been changed from (-inf to + inf) to (0 to + inf).
integrand_1 = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2),len(frequency)))
idx = 0
for idx in np.arange(int(len(IFOs) * (len(IFOs) - 1) / 2)):
    integrand_1[idx,:] =  frequency**-3 * omega_gw * gamma[idx,:] * of_JH[idx,:]
    idx += 1

integrand_1[np.isinf(integrand_1)] = 0
integrand_1[np.isnan(integrand_1)] = 0
# print('integrand_1',integrand_1)
# print(np.shape(integrand_1))

mu = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), 1))
idx = 0
for idx in np.arange(int(len(IFOs) * (len(IFOs) - 1) / 2)):
    mu[idx,:] = 2 * (3. * H0 ** 2)/(20 * np.pi ** 2) * T_seg * np.sum(integrand_1[idx,:]) * (frequency[1] - frequency[0])
    idx += 1

# print('Mean value', mu)
# print(np.shape(mu))

## variance of cross-correaltion signal
integrand_2 = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), len(frequency)))
idx = 0
for idx in np.arange(int(len(IFOs) * (len(IFOs) - 1) / 2)):
    integrand_2[idx,:] = psd[0,:] * psd[1,:] * of_JH[idx,:]**2
    idx += 1

integrand_2[np.isinf(integrand_2)] = 0
integrand_2[np.isnan(integrand_2)] = 0
# print('integrand_2',integrand_2)
# print(np.shape(integrand_2))

sigma = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), 1))
idx = 0
for idx in np.arange(int(len(IFOs) * (len(IFOs) - 1) / 2)):
    sigma[idx,:] = 2 * (T_seg / 4) * np.sum(integrand_2[idx,:]) * (frequency[1] - frequency[0])
    idx += 1

# print('Standard Deviation value', sigma)

# ## Cross-correlation of gravitaional wave strain. H_cc is fourier transform of cross-corelation of the output of two detectors.
# # i.e. gravitaional strain h1 and h2 of detector 1 and 2 respectively.
# H_cc = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), len(frequency)))
# idx = 0
# for idx in np.arange(int(len(IFOs) * (len(IFOs) - 1) / 2)):
#     H_cc =  (3. * H0 ** 2)/(20* np.pi **2) * frequency**-3 * omega_gw * gamma[idx,:]
#     idx += 1
#
# H_cc[np.isinf(H_cc)] = 0
# H_cc[np.isnan(H_cc)] = 0
# print('H_CC is',H_cc)
#
# ## Maximization of SNR signal after using the Optimal Filter
# SNR = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), 1))
# idx = 0
# for idx in np.arange(int(len(IFOs) * (len(IFOs) - 1) / 2)):
#     SNR[idx,:] = (mu[idx,:]/ sigma[idx,:])
#     idx += 1
#
# print('SNR', SNR)
#
# ## SNR for optimal filter Jan_paper
# SNR = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), 1))
# idx = 0
# for idx in np.arange(int(len(IFOs) * (len(IFOs) - 1) / 2)):
#     SNR = np.sqrt(np.sum(gamma[idx,:]**2 * Sb**2)/np.sum(psd[0,:]*psd[1,:]))
#     idx += 1
#
# SNR[np.isinf(SNR)] = 0
# SNR[np.isnan(SNR)] = 0
# print('SNR', SNR)

# def snr(frequency):
#     return  np.sqrt(np.sum(gamma ** 2 * Sb ** 2) / (psd[0, :] * psd[1, :]))
#
# SNR = snr(frequency)
# SNR[np.isinf(SNR)] = 0
# SNR[np.isnan(SNR)] = 0
# print('SNR is',SNR)
#
# plt.figure(3)
# plt.plot(frequency, snr(frequency))
# plt.xlabel('f (Hz)')
# plt.ylabel(r'SNR')
# plt.savefig('SNR_Jan_paper')
