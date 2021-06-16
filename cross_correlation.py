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
from scipy.fftpack import fft, rfft, fftfreq, rfftfreq
from scipy.signal import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

from astropy.cosmology import FlatLambdaCDM

current_direc = os.getcwd()
print("Current Working Directory is :", current_direc)

## Specify the output directory and the name of the simulation.
outdir = 'outdir10'
label = 'injected_signal'
label1 = 'subtracted_signal'
label2 = 'projected_signal'

bilby.utils.setup_logger(outdir=outdir, label=label)

ifos = ['CE', 'ET_D_TR']

## Sampling frequency/Rate. The number of samples per second (in Hz).
sampling_frequency = 2048.

## Set the duration and sampling frequency of the data segment that we're going to inject the signal into
start_time = 1198800017  # for one year beginning
end_time = 1230336017  # for one year long run.

## Total Duration for data taking is
duration = end_time - start_time
print('Total Duration is {}'.format(duration))

## Divide the duration in number of segments i.e. n_seg.
#number of time segment
n_seg = 10000

# Duration of each segment is
duration_seg = duration / n_seg
print('Duration of one segment (duration_seg) is {}'.format(duration_seg))

## Converting duration of each segment into bits.
duration_seg = 2**(int(duration_seg)-1).bit_length()
print('Duration of one segment (duration_seg) in bits is {}'.format(duration_seg))

## Number of samples in each segment of duration (i.e. duration_seg) are
n_samples = int(sampling_frequency*duration_seg/2)+1
print('Number of Samples in each duration_seg are {}'.format(n_samples))

# Number of truncated time Segment are
n_seg = np.trunc(duration/duration_seg)
print('Number of truncated time segment are {}'.format(n_seg))

## load the injections
injections = deepdish.io.load('injections_test.hdf5')['injections']
# print(injections)

## Total number of injection signals
n_inj = len(injections)
# print('Total Number of Injections  is :', n_inj)

## Fixed arguments passed into the source model
waveform_arguments = dict(waveform_approximant='IMRPhenomPv2', reference_frequency=50.,
                          minimum_frequency=2.)

## Create the waveform_generator using a LAL BinaryBlackHole source function
## set duration = duration_seg. for duration =4. there will be an error.
waveform_generator = bilby.gw.WaveformGenerator(duration=duration_seg, sampling_frequency=sampling_frequency,
                                                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                                                waveform_arguments=waveform_arguments)

## Subtraction Parameters ##
'''defining a function to read all max_likelihood from all sample datafiles and save them as new DataFrame.
   subtraction_parameters used here are the maximum likelihood parameters.'''

def multi_det_subtracted_parameters():
    ## reading the max likelihood directory
    multi_det_max_likelihood = os.path.join(current_direc, 'multi_det_max_likelihood')
    multi_det_max_likelihood_data = os.listdir('multi_det_max_likelihood')
    multi_det_max_likelihood_data_select = [x for x in multi_det_max_likelihood_data if x.startswith('37')]

    ## for List of Keys.
    multi_det_subtracted_param_list = []
    ## For making a dict of all values and their keys.
    multi_det_subtracted_params_dict = []

    i = 0

    for multi_det_max_likelihood_data_direc in sorted(multi_det_max_likelihood_data_select):
        if i < 100:
            multi_det_current_max_likelihood_data_direc = os.path.join(multi_det_max_likelihood, multi_det_max_likelihood_data_direc)

            multi_det_max_likelihood_sample_data_file_name = os.path.join(multi_det_current_max_likelihood_data_direc,
                                                                'sample_param_' + str(i) + '_result.json')

            ## Loading the datafile in reading mode.
            multi_det_max_likelihood_sample_data_file_open = json.load(open(multi_det_max_likelihood_sample_data_file_name))

            multi_det_max_likelihood_sample_data_file_open_read = multi_det_max_likelihood_sample_data_file_open['posterior']['content']

            key_list =  []
            dict ={}
            for sub_sub_keys in multi_det_max_likelihood_sample_data_file_open_read:
                # print('sub_sub_keys in sub_key content of main_key posterior of json data_file', sub_sub_keys)
                # print((multi_det_max_likelihood_sample_data_file_open_read [sub_sub_keys][-1]))

                key_list.append(sub_sub_keys)
                dict[sub_sub_keys] = multi_det_max_likelihood_sample_data_file_open_read[sub_sub_keys][-1]

            ## Deleting the maximum log_likelihood and log_prior parameter from multi_det_max_likelihood_sample_data_file_open_read .
            del (dict['log_likelihood'])
            del (dict['log_prior'])

            multi_det_subtracted_param_list.append(key_list)
            multi_det_subtracted_params_dict.append(dict)

            i += 1

    return multi_det_subtracted_params_dict

## Subtraction Parameters are
subtraction_param = multi_det_subtracted_parameters()
#print(subtraction_param)

## Changing 'iota' to 'theta_jn' to be suitable with bilby
#subtraction_param['theta_jn'] = subtraction_param.pop('iota')

## Total Number of Subtraction/estimated signals are
sub_inj = len(subtraction_param) # number of subtraction parameters.
print('sub_inj',sub_inj)

## Number of parameters in one subtracted/estimated signal are
n_params = len(subtraction_param[-1])
print('n_params',n_params)

## Changing default n_seg and n_inj
n_seg = 100   # time segment
n_inj = 100   # number of Injection signals

IFOs = bilby.gw.detector.InterferometerList(ifos)

#################################
''' Overlap Reduction Function'''
#################################

## Nquist Frequency i.e. maximum frequency that can be analyzed.
Nquist = sampling_frequency / 2

## Number of samples in duration_seg or number of FFT bins
N_samples = int(sampling_frequency * duration_seg)  # int(sampling_frequency*duration_seg/2)+1
print('N_samples',N_samples)

## sampling interval
delta_T = duration_seg / N_samples
print('delta_T',delta_T)

## frequency resolution/ FFT bin resolution i.e. frequency per bin of FFT
frequency_resolution = 1 / duration_seg ## frequency_resolution in general is FR = sampling_frequency/(N_samples) = 1/duration_seg
print('frequency_resolution',frequency_resolution)

# ## FFT bins
# FFT_bins = N_samples/2
# print('FFT_bins',FFT_bins)
# FFT_bins_rsolution = Nquist/FFT_bins
# print('FFT_bins_rsolution',FFT_bins_rsolution)

## dividing the duration_seg into sub segments
## number of segments into which we want to divide
n_T_seg = 512

## duration of each segment
T_seg = duration_seg / n_T_seg  # T = 8 sec
T_seg = 2 ** (int(T_seg) - 1).bit_length()
print('T_seg', T_seg)

## sampling Interval i.e. spacing between two T_seg points.
delta_T_seg = T_seg / sampling_frequency
print('delta_T_seg',delta_T_seg)

# ## frequency bins for n_samples.
# n_frequencies = int(np.round(n_samples / 2) + 1)
# print('n_frequencies',n_frequencies)
#
# freq_series = np.linspace(start=0, stop=sampling_frequency / 2, num=n_frequencies)
# print('freq_series',freq_series)
# print(len(freq_series))
# ## frequency bin resolution for number of bins is
# # n_freq_resl = nquist/n_frequencies

## total number of samples in T_seg of FFT bins
n_samples_fft = int(np.round(T_seg * sampling_frequency))
print('n_samples_fft',n_samples_fft)

## frequency resolution/ FFT bin resolution
FFT_resolution = sampling_frequency / n_samples_fft
print('FFT_resolution',FFT_resolution)

## frequency bins for n_samples.
n_frequencies = int(np.round(n_samples_fft/2) + 1)
print('n_frequencies',n_frequencies)

freq_series = np.linspace(start=0, stop=sampling_frequency / 2, num=n_frequencies)
print('freq_series',freq_series)
print(len(freq_series))

## frequency bin resolution for number of bins is
n_freq_resl = Nquist/n_frequencies
print('n_freq_resl',n_freq_resl)

## rfftfreq : Return the real Discrete Fourier Transform sample frequencies.
freq_rfft = rfftfreq(n_samples_fft, d=1 / sampling_frequency)
print('freq_rfft',freq_rfft)

n_freq_resl = len(freq_rfft)
print('n_freq_resl',n_freq_resl)

frequency = freq_rfft

modes = ['plus', 'cross']  # ,'breathing']

## Calculate the position of the detector vertex in geocentric coordinates in meters.
position = np.zeros((len(IFOs), 3))
cp = 0
for ifo in IFOs:
    position[cp, :] = ifo.vertex_position_geocentric()
    cp += 1

### Antena Response/Pattern factor  for a given detector at given time and angles
'''
phi = ra:  right ascension in radians 
theta = dec: declination in radians
'''
ra_vec = np.linspace(0, 2 * np.pi, 100)
dec_vec = np.linspace(-np.pi / 2, np.pi / 2, 100)

## d_omega = sin(theta) * d_theta * d_phi.
d_dec = dec_vec[1] - dec_vec[0]
d_ra = ra_vec[1] - ra_vec[0]
d_sin = np.sin(dec_vec[1]) - np.sin(dec_vec[0])

[ra_mat, dec_mat] = np.meshgrid(ra_vec, dec_vec)
ra_vec = ra_mat.flatten()
dec_vec = dec_mat.flatten()
''' 
F_+ and F_× will depend on the three angles defining the sky position and polarization basis.
using equatorial co¨ordinates, they will depend on right ascension, declination and the polarization angle ψ.
Note antenna_response in bilby.gw.antenna_response gives  dot product between detector_tensor and polarization tensor.
'''
antenna_response = np.zeros((len(IFOs), len(modes), len(ra_vec)))
ci = 0
for ifo in IFOs:
    cm = 0
    for mode in modes:
        pidx = 0
        for pidx in range(len(ra_vec)):
            antenna_response[ci, cm, pidx] = ifo.antenna_response(ra_vec[pidx], dec_vec[pidx], 0, 0, mode)
            # print('antenna_response', antenna_response)
            pidx += 1
        cm += 1
    ci += 1

def odf(frequency):

    orf = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), len(frequency)))
    eins = np.ones(len(frequency))
    di = 0
    ## Iterate Over first detectors
    for detector_1 in range(len(IFOs)):
        f1p = antenna_response[detector_1, 0, :]  # mode = 0 i.e. plus polarisation
        f1c = antenna_response[detector_1, 1, :]  # mode = 1 i.e. cross polarisation
        ## Iterate Over second detectors
        for detector_2 in range(detector_1 + 1, len(IFOs)):
            f2p = antenna_response[detector_2, 0, :]
            f2c = antenna_response[detector_2, 1, :]

            plus_mode = np.einsum('i,i->i', f1p, f2p)
            cross_mode = np.einsum('i,i->i', f1c, f2c)
            total_response = np.einsum('i,i->i', plus_mode, cross_mode)

            delta_x = position[detector_1, :] - position[detector_2, :]
            omega = np.array([np.cos(dec_vec) * np.cos(ra_vec), np.cos(dec_vec) * np.sin(ra_vec), np.sin(dec_vec)])

            orf[di, :] = (5 / (8 * np.pi)) * np.sum(
                np.outer(eins, np.cos(dec_vec) * (f1p * f2p + f1c * f2c)) * np.exp(1j * 2 * np.pi *
                    np.outer(frequency, np.dot(omega.T, delta_x)) / speed_of_light), axis=1) * d_dec * d_ra

            di += 1

    return orf

gamma = odf(frequency)
plt.figure(1)
plt.plot(frequency, gamma[0, :], label = 'CE & ET_D_TR_1')
plt.plot(frequency, gamma[1, :], label = 'CE & ET_D_TR_2')
plt.plot(frequency, gamma[2, :], label = 'CE & ET_D_TR_3')
# plt.plot(frequency, gamma[3, :], label = 'ET_D_TR_1 & ET_D_TR_2')
# plt.plot(frequency, gamma[4, :], label = 'ET_D_TR_1 & ET_D_TR_3')
# plt.plot(frequency, gamma[5, :], label = 'ET_D_TR_2 & ET_D_TR_3')
legend = plt.legend(loc='lower right', fontsize='small')
# plt.xscale('log')
plt.xlim(0, 600)
# plt.axhline(y=0)
# plt.axvline(x=0)
plt.xlabel(r'f (Hz)')
plt.ylabel(r'gamma(f)')
plt.title(r'Overlap Reduction Function Between CE and ET_D_TR')
plt.savefig('Overlap Reduction Function')
# plt.savefig('./outdir10/Overlap Reduction Function_log')
plt.show()
plt.close()

#################################
''' Optimal Signal Processing '''
#################################
# SGWB and noise intrinsic to detector are stationary.
# Cross-correlation signal between the output of two detectors

## Set One sided Noise Power Spectral Density of the detectors (because frequecy is from 0 to higher values).
psd = np.zeros((len(IFOs), len(frequency)))
ci = 0
for ifo in IFOs:
    PSD = ifo.power_spectral_density
    psd[ci,] = PSD.power_spectral_density_interpolated(frequency)
    ci += 1

psd[np.isinf(psd)] = 0
psd[np.isnan(psd)] = 0

## Spectrum of Stochastic Gravitaional wave background
## Hubble constant H0 = (67.4±0.5) km s−1Mpc−1
G = 6.67408 * 10 ** -11  ## units = m**3/ (kg* sec**2)
one_pc = 3.0856775814671916 * 10 ** 16  ## units = m
H0 = 67.4 * 10 ** 3 * 10 ** -6 * one_pc ** -1  ## units = 1/sec

## rho_c = 3 * c**2 * H0**2/(8 * np.pi* G)
rho_c = (3 * speed_of_light ** 2 * H0 ** 2) / (8 * np.pi * G)  ## units = erg/cm**3
omega_gw = 10 ** -15

## Optimal filter depends upon the location and orientation of detector as well as SGWB and Noise PSD of detector.
## Allen and Romano 1999 Eq. 3.73
def optimal_filter_cal(frequency):

    optimal_filter = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), len(frequency) ))
    cidx = 0
    for idx in np.arange((int(len(IFOs) * (len(IFOs) - 1) / 2))):

        Sb = np.dot((3. * H0 ** 2) / (4 * np.pi ** 2), np.divide(omega_gw, frequency ** 3))
        Sb[np.isinf(Sb)] = 0

        optimal_filter[cidx, :] = (gamma[idx,:] * Sb) / (psd[0, :] * psd[1, :])
        cidx += 1

    return optimal_filter

optimal_filter = optimal_filter_cal(frequency)
optimal_filter[np.isinf(optimal_filter)] = 0
optimal_filter[np.isnan(optimal_filter)] = 0

plt.figure(2)
plt.plot(frequency, optimal_filter[0, :],  label = 'CE & ET_D_TR_1')
plt.plot(frequency, optimal_filter[1, :],  label = 'CE & ET_D_TR_2')
plt.plot(frequency, optimal_filter[2, :],  label = 'CE & ET_D_TR_3')
# plt.plot(frequency, optimal_filter[3, :],  label = 'ET_D_TR_1 & ET_D_TR_2')
# plt.plot(frequency, optimal_filter[4, :],  label = 'ET_D_TR_1 & ET_D_TR_3')
# plt.plot(frequency, optimal_filter[5, :],  label = 'ET_D_TR_2 & ET_D_TR_3')
legend = plt.legend(loc='lower right', fontsize='small')
# plt.xscale('log')
plt.xlim(0, 150)
plt.xlabel(r'f(Hz)')
plt.ylabel(r'Optimal Filter(f)')
plt.title(r'Optimal Filter CE and ET_D_TR')
plt.savefig(r'Optimal Filter CE and ET')
# plt.savefig(r'./outdir10/Optimal Filter CE and ET_D_TR_log')
plt.show()
plt.close()

for k in np.arange(n_seg):

    # print(' Number of time segment is : ', k)

    t0 = start_time + k * duration_seg

    t1 = t0 + duration_seg

    IFOs.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=duration_seg, start_time=t0)

    #################################
    # injection Parameter and Signal#
    #################################

    injected = False

    for j in np.arange(n_inj):

        # print("Number of Injection signal is :", j)

        injection_parameters = dict(injections.loc[j])
        # print("Injection Parameters are : ", injection_parameters)

        ## Changing 'iota' to 'theta_jn' to be suitable with bilby
        injection_parameters['theta_jn'] = injection_parameters.pop('iota')

        ## check for merger time and injection time segment range.
        geocent_time = injection_parameters['geocent_time']
        # print("Geocent Time is : ",geocent_time)

        if t0 < injection_parameters['geocent_time'] < t1:

            # print('Calculating Injection Parameter Signal')

            injected = True

            print(' Number of time segment is : ', k)

            print("Checked")

            print("Time t0 is : ", t0)
            print('Time t1 is :', t1)
            print("Geocent Time is : ", geocent_time)

            print('t0 is less than injection_parameter_geocent_time which is less than t1')
            # print("t0 = {} is less than injection_parameter_geocent_time = {} which is less than t0+duration_seg = {} ".format(t0, injection_parameters['geocent_time'], t1))

            print("Number of Injection signal is :", j)
            # print("Injection Parameters for Injection Signal {} are {}  ".format(j, injection_parameters))

            ## Redshift to luminosity Distance conversion using bilby
            injection_parameters['luminosity_distance'] = bilby.gw.conversion.redshift_to_luminosity_distance(
                injection_parameters['redshift'])

            ## First mass needs to be larger than second mass
            if injection_parameters['mass_1'] < injection_parameters['mass_2']:
                tmp = injection_parameters['mass_1']
                injection_parameters['mass_1'] = injection_parameters['mass_2']
                injection_parameters['mass_2'] = tmp

            alpha_1 = injection_parameters

            injected_signal = IFOs.inject_signal(parameters=injection_parameters, injection_polarizations=None,
                                                 waveform_generator=waveform_generator)

            # print(injected_signal)

            '''
            ## read injected_signal (list of multiple dictionaries.)
            for item in injected_signal:
                print(item)
                for key in item:
                    print(key)
                    print(item[key])
            '''

            ##################
            # Cross_Corelation
            ##################

            ## calculating time_doamin_strain for all detectors.
            strain_data = np.zeros((len(IFOs), int(N_samples)))
            ci = 0
            for ifo in IFOs:
                strain_data[ci, :] = ifo.strain_data.time_domain_strain
                print('strain_data', strain_data)
                ci += 1
            print('strain_data', strain_data)
            print(np.shape(strain_data))

            # # If time domain data is not available the one can generate time domain data from frequency domain strain
            # strain_data = np.zeros((len(IFOs), int(N_samples/2)+1)) ## N_samples = (Nquist * duration_seg)+1
            # ci = 0
            # for ifo in IFOs:
            #     strain_data[ci, :] = ifft(ifo.strain_data.frequency_domain_strain)
            #     print('strain_data', strain_data)
            #     ci += 1
            # print('strain_data', strain_data)
            # print(np.shape(strain_data))

            ## Calculating the cross corelation between two detectors.
            cross_correlation = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), len(frequency)))
            cifo =0
            for ifo1 in range(len(IFOs)):
                detector_1 = strain_data[ifo1,:] #ifo1.strain_data.frequency_domain_strain
                # print(detector_1)
                # print(np.shape(detector_1))
                for ifo2 in range(ifo1 + 1, len(IFOs)):
                    detector_2 = strain_data[ifo2,:] #ifo2.strain_data.frequency_domain_strain
                    # print(detector_2)
                    # print(np.shape(detector_2))
                    # print(np.shape(optimal_filter[cifo,:]))

                    freq_cc, cross_correl = scipy.signal.csd(detector_1,
                                                            np.conj(detector_2), fs =sampling_frequency, nperseg =  n_samples_fft)
                    # print('freq_cc',freq_cc)
                    # print(np.shape(freq_cc))
                    # print('cross_correl is', cross_correl)
                    # print(np.shape(cross_correl))

                    plt.figure(k+cifo+ifo2+4)
                    plt.plot(freq_cc, cross_correl, label='CE & ET_D_TR_'+str(k)+str(j)+str(cifo)+str(ifo2))
                    legend = plt.legend(loc='lower right', fontsize='small')
                    # plt.xscale('log')
                    plt.xlim(1, 1000)
                    plt.xlabel(r'f (Hz)')
                    plt.ylabel(r'Cross Correlation')
                    plt.title(r'Cross Correlation Between CE and ET_D_TR_'+str(k)+str(j)+str(cifo)+str(ifo2))
                    plt.savefig('./outdir10/Cross Correlation Function_'+str(k)+str(j)+str(cifo)+str(ifo2)+'.png')
                    plt.show()
                    plt.close()

                    cross_correlation[cifo,:] = np.sum(np.real(np.outer(cross_correl, optimal_filter[cifo,:]))) * (frequency[1]-frequency[0])
                    print('cross_correlation', cross_correlation)
                    print(np.shape(cross_correlation))

                    cifo += 1

            print('cross_correlation',cross_correlation)
            print(np.shape(cross_correlation))

            for idx in np.arange(6):
                plt.figure(idx+11)
                plt.plot(frequency, cross_correlation[idx, :], label='Cross Corelation between CE & ET_D_TR_1' + str(idx)+str(k) + str(j))
                legend = plt.legend(loc='lower right', fontsize='small')
                # plt.xscale('log')
                # plt.xlim(0, 150)
                plt.xlabel(r'f(Hz)')
                plt.ylabel(r'Cross-Correlation')
                plt.title(r'Cross Corelation between CE and ET_D_TR')
                plt.savefig(r'./outdir10/Cross Corelation between CE and ET' + str(idx) + str(k) + str(j))
                plt.show()
                plt.close()


            ## Expectationv alue of corelation function is
            cross_correlation_expect = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), len(frequency)))
            ## variance of correlation noise is
            cross_correlation_var = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), len(frequency)))

            cidx = 0
            for idx in np.arange((int(len(IFOs) * (len(IFOs) - 1) / 2))):
                ## Sb is GW strain power spectral density
                Sb = np.dot((3. * H0 ** 2) / (4 * np.pi ** 2), np.divide(omega_gw, frequency ** 3))
                Sb[np.isinf(Sb)] = 0

                cross_correlation_expect[cidx,:] = np.sum(gamma[idx,:] * Sb * optimal_filter[idx,:]) * (frequency[1]-frequency[0])
                cross_correlation_var[cidx,:] = np.sum(psd[0,:] * psd[1,:] * optimal_filter[idx,:]**2) * (frequency[1]-frequency[0])

                cidx += 1

            print('cross_correlation_expect',cross_correlation_expect)
            print(np.shape(cross_correlation_expect))
            print('cross_correlation_var',cross_correlation_var)
            print(np.shape(cross_correlation_var))


        # else:
        #    print("Above Condition for Injection Parameters is Not Satisfied")

    if injected:
        label = 'inj_segment_' + str(k)
        IFOs.save_data(outdir=outdir, label=label)
        IFOs.plot_data(outdir='./outdir10', label=label)
