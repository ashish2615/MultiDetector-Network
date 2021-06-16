from __future__ import division, print_function

import os
import sys
import bilby
import deepdish
import numpy as np
import logging
import deepdish
import pandas as pd
import json
import math
import sklearn
import seaborn as sns

from bilby.core.utils import speed_of_light

import scipy
from scipy import signal, fftpack
from scipy.fftpack import fft, rfft, fftfreq, rfftfreq
from scipy.signal import (periodogram, welch, lombscargle, csd, coherence,
                          spectrogram)
from scipy.signal import *

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')

from astropy.cosmology import FlatLambdaCDM

current_direc = os.getcwd()
print("Current Working Directory is :", current_direc)

## Specify the output directory and the name of the simulation.
outdir = 'outdir00'
label = 'injected_signal'
label1 = 'subtracted_signal'
label2 = 'projected_signal'
bilby.utils.setup_logger(outdir=outdir, label=label)

from overlap_reduction_function import frequency, odf, N_samples, n_samples_fft
from Optimal_filter import optimal_filter_calculation, H0, omega_gw, psd

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

            '''
            ## read injected_signal (list of multiple dictionaries.)
            for item in injected_signal:
                print(item)
                for key in item:
                    print(key)
                    print(item[key])
            '''

            ###########################################################
            '''Cross Correlation of Detector Channels for Injection'''
            ###########################################################

            ## calculating time_doamin_strain for all detectors.
            strain_data = np.zeros((len(IFOs), int(N_samples)))
            ci = 0
            for ifo in IFOs:
                strain_data[ci, :] = ifo.strain_data.time_domain_strain
                print('strain_data', strain_data)
                ci += 1
            print('strain_data', strain_data)
            print(np.shape(strain_data))

            # # If time domain data is not available, generate time domain data from frequency domain strain
            # strain_data = np.zeros((len(IFOs), int(N_samples/2)+1)) ## N_samples = (Nquist * duration_seg)+1
            # ci = 0
            # for ifo in IFOs:
            #     strain_data[ci, :] = ifft(ifo.strain_data.frequency_domain_strain)
            #     print('strain_data', strain_data)
            #     ci += 1
            # print('strain_data', strain_data)
            # print(np.shape(strain_data))

            ## Calculating the cross corelation between two detectors.
            cross_correlation = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), len(frequency)), dtype=np.complex)
            cifo = 0
            for ifo1 in range(len(IFOs)):
                detector_1 = strain_data[ifo1, :]  # ifo1.strain_data.frequency_domain_strain
                # print(detector_1)
                # print(np.shape(detector_1))
                for ifo2 in range(ifo1 + 1, len(IFOs)):
                    detector_2 = strain_data[ifo2, :]  # ifo2.strain_data.frequency_domain_strain
                    # print(detector_2)
                    # print(np.shape(detector_2))

                    freq_cc, cross_correl = scipy.signal.csd(detector_1, detector_2, fs=sampling_frequency,
                                                             nperseg=n_samples_fft)

                    # print('freq_cc', freq_cc)
                    # print(np.shape(freq_cc))
                    # print('cross_correl is', cross_correl)
                    # print(np.shape(cross_correl))

                    cross_correlation[cifo, :] = (cross_correl * optimal_filter[cifo, :]) * (
                                frequency[1] - frequency[0])
                    # cross_correlation[cifo, :] = np.sum(np.real(cross_correl * optimal_filter[cifo, :])) * (frequency[1] - frequency[0])

                    cifo += 1

            print('cross_correlation', cross_correlation)
            print(np.shape(cross_correlation))

            for idx in np.arange(6):
                plt.figure(idx)
                plt.plot(frequency, np.real(cross_correlation[idx, :]),
                         label='Injected Cross Corelation between CE & ET_D_TR_1' + str(k) + str(j)+ str(idx))
                legend = plt.legend(loc='lower right', fontsize='small')
                # plt.xscale('log')
                plt.xlim(0, 150)
                plt.xlabel(r'f(Hz)')
                plt.ylabel(r'Cross-Correlation')
                plt.title(r'Injected Cross Corelation between CE and ET_D_TR')
                plt.savefig(r'./outdir00/Injected Cross Corelation between CE and ET'  + str(k) + str(j)+ str(idx))
                plt.show()
                plt.close()

            ## Sb is GW strain power spectral density
            Sb = np.dot((3. * H0 ** 2) / (4 * np.pi ** 2), np.divide(omega_gw, frequency ** 3))
            Sb[np.isinf(Sb)] = 0

            ## Expectation value of corelation function is
            cross_correlation_expect = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), 1))
            cidx = 0
            for idx in np.arange((int(len(IFOs) * (len(IFOs) - 1) / 2))):

                cross_correlation_expect[cidx, :] = np.sum(gamma[idx, :] * Sb * optimal_filter[idx, :]) * (
                            frequency[1] - frequency[0])
                # print('cross_correlation_expect[cidx, :]', (cross_correlation_expect[cidx, :]))

                cidx += 1

            print('cross_correlation_expect', cross_correlation_expect)
            print(np.shape(cross_correlation_expect))

            ## variance of correlation noise is
            cross_correlation_var = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), 1))
            cidx = 0
            for idx in np.arange((int(len(IFOs) * (len(IFOs) - 1) / 2))):

                cross_correlation_var[cidx, :] = np.sum(psd[0, :] * psd[1, :] * optimal_filter[idx, :] ** 2) * (
                            frequency[1] - frequency[0])

                # print(cross_correlation_var[cidx, :])

                cidx += 1

            print('cross_correlation_var', cross_correlation_var)
            print(np.shape(cross_correlation_var))

            ## optimal estimator is given as
            ## Eq. 6 (PRL 113, 231101 (2014))
            # optimal_est = np.array([])
            optimal_estimator = np.zeros((1))
            for idx in np.arange((int(len(IFOs) * (len(IFOs) - 1) / 2))):
                optimal_estimator += np.sum(
                    np.real(cross_correlation[idx, :]) * cross_correlation_var[idx, :] ** -2) / np.sum(
                    cross_correlation_var[idx, :] ** -2)

            print('optimal_estimator', optimal_estimator)

    # else:
        #    print("Above Condition for Injection Parameters is Not Satisfied")

    if injected:
        label = 'inj_segment_' + str(k)
        IFOs.save_data(outdir=outdir, label=label)
        IFOs.plot_data(outdir='./outdir00', label=label)

    ##############################
    # Sample Parameter and Signal#
    ##############################

    subtracted = False

    for x in np.arange(sub_inj):

        # print("Number of subtraction signal is :", x)

        ## convertind subtraction_param DataFrame to dictionary
        subtraction_parameters = dict(subtraction_param[x])

        ## Change the luminosity_distance for reducing the signal from Noise of detector.
        subtraction_parameters['luminosity_distance'] = float(subtraction_parameters['luminosity_distance'])

        ## check for merger time and subtraction time segment range.
        geocent_time = subtraction_parameters['geocent_time']

        if t0 < subtraction_parameters['geocent_time'] < t1:

            subtracted = True

            print(' Number of time segment is : ', k)

            print("Checked")

            print("Time t0 is : ", t0)
            print('Time t1 is :', t1)
            print("Geocent Time is : ", geocent_time)

            print('t0 is less than subtraction_parameter_geocent_time which is less than t1')
            # print("t0 = {} is less than subtraction_parameter_geocent_time = {} which is less than t0+duration_seg = {} ".format(t0, subtraction_parameters['geocent_time'], t1))

            print("Number of Subtraction signal is :", x)
            print("Subtraction parameters for subtraction signal {} are {} ".format(x, subtraction_parameters))

            ## First mass needs to be larger than second mass (just to cross check)
            if subtraction_parameters['mass_1'] < subtraction_parameters["mass_2"]:
                tmp = subtraction_parameters['mass_1']
                subtraction_parameters['mass_1'] = subtraction_parameters['mass_2']
                subtraction_parameters['mass_2'] = tmp

            alpha_2 = subtraction_parameters

            subtracted_signal = IFOs.subtract_signal(parameters=subtraction_parameters,
                                                     injection_polarizations=None,
                                                     waveform_generator=waveform_generator)  # to be derived from bilby's inject_signal

            frequency_strain = waveform_generator.frequency_domain_strain(subtraction_parameters)
            # print('Frequency_Domain_strain is', frequency_strain)
            frequency_array = waveform_generator.frequency_array
            # print('Frequency Array is :', frequency_array)

            ##########################
            ''' Residual Data Noise'''
            ##########################

            ## Detector Residual Data for each Sample After Subtracting the Signal
            residual_noise_data = np.zeros((len(IFOs), n_samples))
            cnt = 0
            for ifo in IFOs:
                residual_noise_data[cnt, :] = ifo.strain_data.frequency_domain_strain
                cnt += 1
                # print('Residual Noise Data is :', residual_noise_data)

            print('Residual Noise Data is :', residual_noise_data)
            print(np.shape(residual_noise_data))


            ############################################################
            '''Cross Correlation of Detector Channels for Subtraction'''
            ############################################################

            # ## calculating time_doamin_strain for all detectors.
            # strain_data = np.zeros((len(IFOs), int(N_samples)))
            # ci = 0
            # for ifo in IFOs:
            #     strain_data[ci, :] = ifo.strain_data.time_domain_strain
            #     print('strain_data', strain_data)
            #     ci += 1
            # print('strain_data', strain_data)
            # print(np.shape(strain_data))

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
            cross_correlation = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), len(frequency)), dtype=np.complex)
            cifo = 0
            for ifo1 in range(len(IFOs)):
                detector_1 = np.real(ifft(residual_noise_data[ifo1, :]))
                # print('d1',detector_1)
                # print(np.shape(detector_1))
                for ifo2 in range(ifo1 + 1, len(IFOs)):
                    detector_2 = np.real(ifft(residual_noise_data[ifo2, :]))
                    # print('d2',detector_2)
                    # print(np.shape(detector_2))

                    freq_cc, cross_correl = scipy.signal.csd(detector_1, detector_2, fs=sampling_frequency,
                                                             nperseg=n_samples_fft)

                    # print('freq_cc',freq_cc)
                    # print(np.shape(freq_cc))
                    # print('cross_correl is', cross_correl)
                    # print(np.shape(cross_correl))

                    cross_correlation[cifo, :] = (cross_correl * optimal_filter[cifo, :]) * (
                                frequency[1] - frequency[0])
                    # cross_correlation[cifo, :] = np.sum(np.real(cross_correl * optimal_filter[cifo, :])) * (frequency[1] - frequency[0])

                    # print('cross_correlation', cross_correlation)
                    # print(np.shape(cross_correlation))

                    cifo += 1

            print('cross_correlation', cross_correlation)
            print(np.shape(cross_correlation))

            for idx in np.arange(6):
                plt.figure(idx)
                plt.plot(frequency, np.real(cross_correlation[idx, :]),
                         label='Subtracted Cross Corelation between CE & ET_D_TR_1' + str(k) + str(x) + str(idx))
                legend = plt.legend(loc='lower right', fontsize='small')
                # plt.xscale('log')
                plt.xlim(0, 150)
                plt.xlabel(r'f(Hz)')
                plt.ylabel(r'Cross-Correlation')
                plt.title(r'Subtracted Cross Corelation between CE and ET_D_TR')
                plt.savefig(r'./outdir00/Subtracted Cross Corelation between CE and ET' + str(k) + str(x) + str(idx))
                plt.show()
                plt.close()

            ## Sb is GW strain power spectral density
            Sb = np.dot((3. * H0 ** 2) / (4 * np.pi ** 2), np.divide(omega_gw, frequency ** 3))
            Sb[np.isinf(Sb)] = 0

            ## Expectation value of corelation function is
            cross_correlation_expect = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), 1))
            cidx = 0
            for idx in np.arange((int(len(IFOs) * (len(IFOs) - 1) / 2))):

                cross_correlation_expect[cidx, :] = np.sum(gamma[idx, :] * Sb * optimal_filter[idx, :]) * (
                        frequency[1] - frequency[0])

                # print('cross_correlation_expect[cidx, :]', (cross_correlation_expect[cidx, :]))

                cidx += 1

            print('cross_correlation_expect', cross_correlation_expect)
            print(np.shape(cross_correlation_expect))

            ## variance of correlation noise is
            cross_correlation_var = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), 1))
            cidx = 0
            for idx in np.arange((int(len(IFOs) * (len(IFOs) - 1) / 2))):

                cross_correlation_var[cidx, :] = np.sum(psd[0, :] * psd[1, :] * optimal_filter[idx, :] ** 2) * (
                        frequency[1] - frequency[0])
                # print(cross_correlation_var[cidx, :])

                cidx += 1

            print('cross_correlation_var', cross_correlation_var)
            print(np.shape(cross_correlation_var))

            ## optimal estimator is given as
            ## Eq. 6 (PRL 113, 231101 (2014))
            # optimal_est = np.array([])
            optimal_estimator = np.zeros((1))
            for idx in np.arange((int(len(IFOs) * (len(IFOs) - 1) / 2))):

                optimal_estimator += np.sum(
                    np.real(cross_correlation[idx, :]) * cross_correlation_var[idx, :] ** -2) / np.sum(
                    cross_correlation_var[idx, :] ** -2)

                # print('optimal_estimator',optimal_estimator)
                # idx += 1

            print('optimal_estimator', optimal_estimator)

            # IFOs.save_data(outdir00=outdir00,label=label1)
            # IFOs.plot_data(outdir00='./outdir1', label=label1)

        # else:
        #    print("Above Condition for Subtracted Parameters is Not Satisfied")

    if subtracted:
        label1 = 'sub_segment_' + str(k)
        IFOs.save_data(outdir=outdir, label=label1)
        IFOs.plot_data(outdir='./outdir00', label=label1)
        proj_data_stream = residual_noise_data

    #############################################
    # Projection Parameter Derivative and Signal#
    #############################################

    projected = False

    for z in np.arange(sub_inj):  # np.arange(sub_inj)

        # print("Number of projected signal is :", z)

        ## convertind subtraction_param DataFrame to dictionary
        projection_parameters = dict(subtraction_param[z])
        # print("Projection Parameters are : ", subtraction_parameters)

        ## Change the luminosity_distance for reducing the signal from Noise of detector.
        projection_parameters['luminosity_distance'] = float(projection_parameters['luminosity_distance'])

        ## check for merger time and projection time segment range.
        geocent_time = projection_parameters['geocent_time']
        # print("Geocent Time is : ",geocent_time)

        if t0 < projection_parameters['geocent_time'] < t1:

            projected = True

            print(' Number of time segment is : ', k)

            print("Checked")

            print("Time t0 is : ", t0)
            print('Time t1 is :', t1)
            print("Geocent Time is : ", geocent_time)

            print('t0 is less than projection_parameter_geocent_time which is less than t1')
            # print("t0 = {} is less than projection_parameter_geocent_time = {} which is less than t0+duration_seg = {} ".format(t0, subtraction_parameters['geocent_time'], t1))

            print("Number of Projection signal is :", z)
            print("Projection parameters for projection signal {} are {} ".format(z, projection_parameters))

            ## First mass needs to be larger than second mass (just to cross check)
            if projection_parameters['mass_1'] < projection_parameters["mass_2"]:
                tmp = projection_parameters['mass_1']
                projection_parameters['mass_1'] = projection_parameters['mass_2']
                projection_parameters['mass_2'] = tmp

            count = 0
            waveform_derivatives = [0 for z1 in range(len(IFOs))]
            for ifo in IFOs:
                ## func defines the signal model which depends on model parameters
                def func(parameters):
                    #print(parameters)
                    polarizations = waveform_generator.frequency_domain_strain(parameters)

                    return ifo.get_detector_response(polarizations, parameters)

                #print("Projection parameters for func are :", func(projection_parameters))

                waveform_derivatives[count] = bilby.core.utils.projection_derivatives(projection_parameters, func)
                count += 1

            derivatives = np.array(waveform_derivatives)
            print("Derivatives are :", derivatives)
            print("Shape of derivatives is :", np.shape(derivatives))

            ## Defining the Fisher Matrix
            fisher_matrices = np.zeros((len(IFOs), n_params, n_params))
            ## Defining the Correlation matrix = Inverse of Fisher Matrix
            correlation_matrices = np.zeros((len(IFOs), n_params, n_params))

            ## Calculation of Fisher Matrix : A scalar product of signal model w.r.t. model parameters.
            ## For every parameter in n_params
            ## iterate through rows
            count = 0
            for ii in range(len(IFOs)):
                #print('ii',ii)
                for q1 in range(n_params):
                    #print('q is {} and n_params are {}'.format(q1,n_params))
                    ## iterate through columns
                    for p1 in range(q1, n_params):
                        #print('p is {}'.format(p1))

                        PSD = ifo.power_spectral_density
                        prod = bilby.gw.utils.inner_product(derivatives[ii, q1], derivatives[ii,p1], waveform_generator.frequency_array, PSD)

                        fisher_matrices[count, q1, p1] = prod
                        fisher_matrices[count, p1, q1] = prod

                correlation_matrices[count, :, :] = np.linalg.inv(fisher_matrices[count, :, :])
                count += 1

            print('Fisher Matrix is :', fisher_matrices)
            print('size of fisher matrix is', np.size(fisher_matrices))
            print('shape of fisher matrix is', np.shape(fisher_matrices))

            print('Co-relation Matrix is :', correlation_matrices)
            print('size of Co-relation matrix is', np.size(correlation_matrices))
            print('shape of Co-relation matrix is', np.shape(correlation_matrices))

            #correlation_matrices_save = np.save('./outdir4/correlation_matrices.npy',correlation_matrices)

            ## Calculating the scalar product of derivatives of data signal w.r.t number of parameters and residual_noise_data
            scalar_product = np.zeros((len(IFOs), n_params))
            #print('Shape of Scalar Product is', np.shape(scalar_product))

            count = 0
            ## Calculation of Scalar Product : A scalar product between two signals and defined by the inner product on the vector space of signals
            ## For every detector in IFOs iteration is
            for detector in range(len(IFOs)):
                ## For every parameter in n_params
                ## iterate through rows
                for q2 in range(n_params):
                    # print('q2 is {}'.format(q2))

                    PSD = ifo.power_spectral_density
                    product = bilby.gw.utils.inner_product(derivatives[detector,q2], residual_noise_data[detector, :],
                                                           waveform_generator.frequency_array, PSD)

                    scalar_product[count, q2] = product

                    proj_data_stream[detector,:] = proj_data_stream[detector,:] - np.matmul(np.matmul(correlation_matrices[detector, :, :], scalar_product[detector,:].T).T, derivatives[detector,:, :])

                count += 1

            print('Scalar Product is ', scalar_product)
            #print('Type of Scalar product is', type(scalar_product))
            print('Shape of Scalar Product is', np.shape(scalar_product))

            print('proj_data_stream is {}'.format(proj_data_stream))
            print('proj_data_stream shape is ', np.shape(proj_data_stream))

            ## Defining the Fisher matrix network for the network of all detectors.
            fisher_matrix_network = np.zeros((n_params, n_params))
            for k in range(len(IFOs)):
                fisher_matrix_network += fisher_matrices[k, :, :]
            print('Fisher_Matrix_Network is ', fisher_matrix_network)
            print('Shape of fisher_matrix_network is', np.shape(fisher_matrix_network))

            ## Normalizing the Fisher Matrix Network.
            fisher_matrix_norm = np.zeros((n_params, n_params))
            for q in range(n_params):
                for p in range(n_params):
                    fisher_matrix_norm[q, p] = fisher_matrix_network[q, p] / np.sqrt(
                        fisher_matrix_network[p, p] * fisher_matrix_network[q, q])
            print('Fisher Matrix Normalization is :', fisher_matrix_norm)
            print('Shape of Fisher Matrix Normalization is',np.shape(fisher_matrix_norm))

            # correlation_matrices_1[z, :, :] = np.linalg.inv(fisher_matrix_norm)
            # print('correlation_matrices for fisher_matrix_network is', correlation_matrices)
            # print('Shape of correlation_matrices for fisher_matrix_network is', np.shape(correlation_matrices))



            # '''## Difference between max_likelihood parameter values and true parameters value ##'''
            #
            # delta_lambda = np.zeros((len(IFOs), n_params))
            # count = 0
            # ## iterating over all detectors
            # for detector in IFOs:
            #     print(detector)
            #     for q3 in range(n_params):
            #
            #         PSD = detector.power_spectral_density
            #         product = bilby.gw.utils.inner_product(detector, derivatives[detector,q3],
            #                                                waveform_generator.frequency_array, PSD)
            #
            #         print('product is', product)
            #
            #         delta_lambda = np.matmul(np.linalg.inv(fisher_matrices[detector,:,:]), product)
            #         delta_lambda[count, q3] = delta_lambda
            #
            #     count += 1

            ###########################################################
            '''Cross Correlation of Detector Channels for Projection'''
            ###########################################################

            ## Calculating the cross corelation between two detectors.
            cross_correlation = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), len(frequency)), dtype=np.complex)
            cifo = 0
            for ifo1 in range(len(IFOs)):
                detector_1 = np.real(ifft(proj_data_stream[ifo1, :]))  # ifo1.strain_data.frequency_domain_strain
                print('detector_1', detector_1)
                # print(np.shape(detector_1))
                for ifo2 in range(ifo1 + 1, len(IFOs)):
                    detector_2 = np.real(ifft(proj_data_stream[ifo2, :]))  # ifo2.strain_data.frequency_domain_strain
                    print('detector_2', detector_2)
                    # print(np.shape(detector_2))
                    # print(np.shape(optimal_filter[cifo,:]))

                    freq_cc, cross_correl = scipy.signal.csd(detector_1, detector_2, fs=sampling_frequency,
                                                             nperseg=n_samples_fft)
                    # print('freq_cc', freq_cc)
                    # print(np.shape(freq_cc))
                    # print('cross_correl is', cross_correl)
                    # print(np.shape(cross_correl))

                    cross_correlation[cifo, :] = (cross_correl * optimal_filter[cifo, :]) * (
                                frequency[1] - frequency[0])
                    # cross_correlation[cifo, :] = np.sum(np.real(cross_correl * optimal_filter[cifo, :])) * (frequency[1] - frequency[0])

                    # print('cross_correlation', cross_correlation)
                    # print(np.shape(cross_correlation))

                    cifo += 1

            print('cross_correlation', cross_correlation)
            print(np.shape(cross_correlation))

            for idx in np.arange(6):
                plt.figure(idx)
                plt.plot(frequency, cross_correlation[idx, :],
                         label='Projection Cross Corelation between CE & ET_D_TR_1' + str(k) + str(z) + str(idx))
                legend = plt.legend(loc='lower right', fontsize='small')
                # plt.xscale('log')
                plt.xlim(0, 150)
                plt.xlabel(r'f(Hz)')
                plt.ylabel(r'Cross-Correlation')
                plt.title(r'Projection Cross Corelation between CE and ET_D_TR')
                plt.savefig(r'./outdir00/Projection Cross Corelation between CE and ET' + str(k) + str(z) + str(idx))
                plt.show()
                plt.close()

            ## Sb is GW strain power spectral density
            Sb = np.dot((3. * H0 ** 2) / (4 * np.pi ** 2), np.divide(omega_gw, frequency ** 3))
            Sb[np.isinf(Sb)] = 0

            ## Expectation value of corelation function is
            cross_correlation_expect = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), 1))
            cidx = 0
            for idx in np.arange((int(len(IFOs) * (len(IFOs) - 1) / 2))):

                cross_correlation_expect[cidx, :] = np.sum(gamma[idx, :] * Sb * optimal_filter[idx, :]) * (
                        frequency[1] - frequency[0])

                # print('cross_correlation_expect[cidx, :]', (cross_correlation_expect[cidx, :]))

                cidx += 1

            print('cross_correlation_expect', cross_correlation_expect)
            print(np.shape(cross_correlation_expect))

            ## variance of correlation noise is
            cross_correlation_var = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), 1))
            cidx = 0
            for idx in np.arange((int(len(IFOs) * (len(IFOs) - 1) / 2))):

                cross_correlation_var[cidx, :] = np.sum(psd[0, :] * psd[1, :] * optimal_filter[idx, :] ** 2) * (
                        frequency[1] - frequency[0])
                # print(cross_correlation_var[cidx, :])

                cidx += 1

            print('cross_correlation_var', cross_correlation_var)
            print(np.shape(cross_correlation_var))

            ## optimal estimator is given as
            ## Eq. 6 (PRL 113, 231101 (2014))
            # optimal_est = np.array([])

            optimal_estimator = np.zeros((1))
            for idx in np.arange((int(len(IFOs) * (len(IFOs) - 1) / 2))):

                optimal_estimator += np.sum(
                    np.real(cross_correlation[idx, :]) * cross_correlation_var[idx, :] ** -2) / np.sum(
                    cross_correlation_var[idx, :] ** -2)
                # idx += 1

            print('optimal_estimator', optimal_estimator)

        # IFOs.save_data(outdir00=outdir00,label=label2)
        # IFOs.plot_data(outdir00='./outdir1', label=label2)

    # else:
    #    print("Above Condition for Projected Parameters is Not Satisfied")

    if projected:

        label1 = 'proj_segment_' + str(k)
        IFOs.save_data(outdir=outdir, label=label2)
        IFOs.plot_data(outdir='./outdir00', label=label2)