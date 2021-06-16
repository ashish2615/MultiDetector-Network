from __future__ import division, print_function

import numpy as np
from numpy import loadtxt

import os
import sys
import bilby
from bilby import prior
import logging
import deepdish
import pandas as pd
import h5py

import math
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

import platform

current_direc = os.getcwd()
print("Current Working Directory is :", current_direc)

## Specify the output directory and the name of the simulation.
outdir = 'outdir10'
label = 'injected_signal'
label1 = 'subtracted_signal'
label2 = 'projected_signal'

bilby.utils.setup_logger(outdir=outdir, label=label)

#ifos = ['CE']

ifos = ['CE', 'ET_D_TR']
print(ifos[0])
print(ifos[1])

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

## Changing default n_seg and n_inj
n_seg = 100   # time segment
n_inj = 100   # number of Injection signals

IFOs = bilby.gw.detector.InterferometerList(ifos)
print(IFOs)

## Define Residual Noise after subtraction
#residual_noise_data = np.zeros((len(IFOs), n_samples))

for k in np.arange(n_seg):

    # print(' Number of time segment is : ', k)

    t0 = start_time + k * duration_seg
    # print('Time t0 is :', t0)
    t1 = t0 + duration_seg
    # print('Time t1 is :',t1)

    IFOs.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=duration_seg
                                                       , start_time=t0)

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

            injected = True

            print(' Number of time segment is : ', k)

            print("Checked")

            print("Time t0 is : ", t0)
            print('Time t1 is :', t1)
            print("Geocent Time is : ", geocent_time)

            print('t0 is less than injection_parameter_geocent_time which is less than t1')
            # print("t0 = {} is less than injection_parameter_geocent_time = {} which is less than t0+duration_seg = {} ".format(t0, injection_parameters['geocent_time'], t1))

            print("Number of Injection signal is :", j)
            print("Injection Parameters for Injection Signal {} are {}  ".format(j, injection_parameters))

            ## Redshift to luminosity Distance conversion using bilby
            injection_parameters['luminosity_distance'] = bilby.gw.conversion.redshift_to_luminosity_distance(
                injection_parameters['redshift'])

            ## First mass needs to be larger than second mass
            if injection_parameters['mass_1'] < injection_parameters['mass_2']:
                tmp = injection_parameters['mass_1']
                injection_parameters['mass_1'] = injection_parameters['mass_2']
                injection_parameters['mass_2'] = tmp

            alpha_1 = injection_parameters

            ## Sky localization parameters
            ra = alpha_1['ra']
            dec = alpha_1['dec']
            psi = alpha_1['psi']
            time = alpha_1['geocent_time']
            modes = ['plus', 'cross','breathing']

            injected_signal = IFOs.inject_signal(parameters=injection_parameters, injection_polarizations=None,
                                             waveform_generator=waveform_generator)

            #########################################
            # Cross Correlation of Detector Channels#
            #########################################

            '''Calculating Overlap Reduction Function for detector network'''

            ## An array of frequencies of GW detector.
            frequency = waveform_generator.frequency_array
            print('Frequency array is ', frequency)

            dict_1 = {}
            polarization_tensor = np.zeros((len(modes)))
            ## Calculate the polarization tensor for a given sky location and time
            count = 0
            for mode in modes:
                print(mode)
                polarization_tensor = bilby.gw.utils.get_polarization_tensor(ra, dec, psi, time, mode)
                print('polarization_tensor for mode {} is {}'.format(mode,polarization_tensor))
                dict_1[mode] = polarization_tensor
                count += 1
            print(dict_1)

            ## Antena Response/Pattern factor  for a given detector at given time and angles
            ra_vec = np.linspace(0, 2 * np.pi, 100)
            dec_vec = np.linspace(-np.pi / 2, np.pi / 2, 100)
            ## d_omega = sin(theta) * d_theta * d_phi.
            d_dec = dec_vec[1] - dec_vec[0]
            d_ra = ra_vec[1] - ra_vec[0]
            d_sin = np.sin(dec_vec[1]) - np.sin(dec_vec[0])

            [ra_mat, dec_mat] = np.meshgrid(ra_vec, dec_vec)
            ra_vec = ra_mat.flatten()
            dec_vec = dec_mat.flatten()

            antenna_response = np.zeros((len(IFOs), len(modes), len(ra_vec)))
            ci = 0
            for ifo in IFOs:
                cm = 0
                for mode in modes:
                    for pind in range(len(ra_vec)):
                        antenna_response[ci, cm, pind] = ifo.antenna_response(ra_vec[pind], dec_vec[pind], 0, 0, mode)
                    cm += 1
                ci += 1

            ## Calculate the position of the detector vertex in geocentric coordinates in meters.
            position = np.zeros((len(IFOs), 3))
            cp = 0
            for ifo in IFOs:
                position[cp, :] = ifo.vertex_position_geocentric()
                cp += 1

            # ## Calculate time delay between two detectors in geocentric coordinates
            # ## time_delay = np.dot(omega, delta_x)/c
            # time_delay_2 = np.zeros((len(IFOs), len(IFOs), len(ra_vec), len(ra_vec)))
            #
            # cd1 = 0
            # for detector_1 in range(len(IFOs)):
            #     cd2 = 0
            #     l = detector_1 + 1
            #     for detector_2 in range(l, len(IFOs)):
            #         cr = 0
            #         for ra in ra_vec:
            #             cd = 0
            #             for dec in dec_vec:
            #                 detector_1_vertex = IFOs[detector_1].vertex
            #                 detector_2_vertex = IFOs[detector_2].vertex
            #                 time_delay_2[cd1, cd2+l, cr, cd] = bilby.gw.utils.time_delay_geocentric(detector_1_vertex, detector_2_vertex, ra, dec, 0)
            #                 cd += 1
            #             cr += 1
            #         cd2 += 1
            #     cd1 += 1
            #
            # print('Time delay between two detectors', time_delay_2)
            # print(np.shape(time_delay_2))

            T = duration_seg
            fs = sampling_frequency
            nquist = fs / 2
            ## Number of samples
            N = T * fs
            ## sampling interval
            delta_T = T / N
            frequency_resolution = 1 / T
            ## or
            # frequency_resolution = fs/N
            ## The frequency resolution does not depend on the length of FFT,
            # but the length of the total sampling time T, i.e. it's 1/T,

            ## dividing the duration into segments
            # n_T_seg = 64
            ## duration of each segment
            # T_seg = T / n_T_seg  # T = 64 sec
            # T_seg = 2 ** (int(T_seg) - 1).bit_length()
            # ## sampling Interval i.e. spacing between two T_seg points.
            # delta_T_seg = T_seg / fs

            ## total number of samples in time domain
            n_samples = int(np.round(T_seg * fs))
            FFT_resolution = fs / n_samples

            ## frequency bins for n_samples.
            n_frequencies = int(np.round(n_samples / 2) + 1)
            freq_series = np.linspace(start=0, stop=fs / 2, num=n_frequencies)
            ## frequency bin resolution for number of bins is
            # n_freq_resl = nquist/n_frequencies

            ## fftfreq : Return the Discrete Fourier Transform sample frequencies.
            # The returned float array f contains the frequency bin centers in cycles per unit of the sample spacing (with zero at the start).
            # For instance, if the sample spacing is in seconds, then the frequency unit is cycles/second.
            # sample spacing = d. Sample spacing is inverse of the sampling rate.
            freq_rfft = rfftfreq(n_samples, d=1 / fs)
            n_freq_resl = len(freq_rfft)
            frequency = freq_rfft

            # orf = np.zeros((len(IFOs) * (len(IFOs) - 1) // 2, len(frequency)))  # len(range(int(
            # eins = np.ones(len(frequency))
            # di = 0
            # ## Iterate Over first detectors
            # for detector_1 in range(len(IFOs)):
            #     # print(' detector_1 is', detector_1)
            #     ## Antenna response for first  detectors
            #     f1p = antenna_response[detector_1, 0, :]
            #     f1c = antenna_response[detector_1, 1, :]
            #     ## Iterate Over second detectors
            #     for detector_2 in range(detector_1 + 1, len(IFOs)):
            #         # print(' detector_2 is', detector_2)
            #         ## Antenna response for second  detectors
            #         f2p = antenna_response[detector_2, 0, :]
            #         f2c = antenna_response[detector_2, 1, :]
            #
            #         ## product of detector response
            #         delta_x = position[detector_1, :] - position[detector_2, :]
            #         omega = np.array(
            #             [np.sin(dec_vec) * np.cos(ra_vec), np.sin(dec_vec) * np.sin(ra_vec), np.cos(dec_vec)])
            #
            #         orf[di, :] = np.sum(np.outer(eins, np.cos(dec_vec) * (f1p * f2p + f1c * f2c)) * np.exp(
            #             1j * 2 * np.pi * np.outer(frequency, np.dot(omega.T, delta_x)) / speed_of_light)) * d_dec * d_ra
            #
            #         di += 1
            #
            # print('Overlap Reduction Function is ', orf)
            # print(np.shape(orf))

            def odf(frequency):
                # sum = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), len(ra_vec)))
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

                        delta_x = position[detector_1, :] - position[detector_2, :]
                        omega = np.array(
                            [np.sin(dec_vec) * np.cos(ra_vec), np.sin(dec_vec) * np.sin(ra_vec), np.cos(dec_vec)])

                        # orf =  np.sum(f1p * f2p + f1c * f2c) * np.exp(
                        #     1j * 2 * np.pi * np.outer(frequency, np.dot(omega.T, delta_x)) / speed_of_light) * d_dec * d_ra * d_sin

                        orf[di, :] = np.sum(np.outer(eins, np.cos(dec_vec) * (f1p * f2p + f1c * f2c)) * np.exp(
                            1j * 2 * np.pi * np.outer(frequency, np.dot(omega.T,
                                                                        delta_x)) / speed_of_light)) * d_dec * d_ra * d_sin

                        di += 1
                return orf

            print('Overlap Reduction Function is ', odf)
            plt.plot(frequency, odf(frequency)[0,:])
            plt.axhline(y=0)
            plt.axvline(x=0)
            plt.xlabel('frequency')
            plt.ylabel(r'\gamma (frequency)')
            plt.show()
            plt.savefig('fft_plot42')

            # ##################
            # # Optimal Filter #
            # ##################
            # c = const.c.to('cm/s') ## cm / sec
            # G = 6.67408e-8 # units	cm^{3} / (gram sec^{2})
            # H_0 = 1.1 + 10^{-28} * c * h_100  ## units (cm^{-1})
            #
            # ## rho_c = (3 * c**2 * H_0 ** 2 )/ (8 * np.pi * G)
            # rho_c =  1.6 * 10 ** {-8} * h_100 ** 2 # units ergs/cm^3
            #
            # ## energy density of gravitaional radiation
            # rho_gw = 1/ rho_c *
            #
            # def rho_gw(frequency):
            #     frequency = np.absolute(frequency)
            #     H = 3 * (H_0 **2 / 32 * np.pi **3) * frequency ** -3 * omega_gw * odf(frequency)
            #
            #
            #     rho_gw = 1/32*np.pi (antenna_response[0,0,:,:]**2 + antenna_response[0,1,:,:]**2)
            #
            #     return (4 * np.pi * c ** 2 / G) * np.sum(rho_gw)
            #
            #     # spectrum of cosmological stochastic background
            # omega_gw = 1 / rho_c *
            #
            # p1 =
            # p2 =
            #
            # def optimal_filter(frequency):
            #
            #     frequency = np.abs(frequency)
            #
            #     psd =
            #     for ifo in IFOs:
            #         if ifo == 'CE'
            #             p1 = ifo.power_spectral_density.get_noise_realisation(self.sampling_frequency, self.duration)
            #         if ifo == 'ET_D_TR_1'
            #             p2 = ifo.power_spectral_density.get_noise_realisation(self.sampling_frequency,
            #                                                                   self.duration)
            #     optimal_filter = odf(frequency) * omega_gw(frequency) / (frequency)**3 * p1 * p2)
            #
            # # for ifo in range(len(IFOs)):
            # #     cross_correlation = np.sum(np.real(residual_noise_data[ifo, :] * np.conj(residual_noise_data[ifo, :])))
            #
            # # IFOs.save_data(outdir=outdir,label=label)
            # # IFOs.plot_data(outdir='./outdir1', label=label)

        # else:
        #    print("Above Condition for Injection Parameters is Not Satisfied")

    if injected:
        label = 'inj_segment_' + str(k)
        IFOs.save_data(outdir=outdir, label=label)
        IFOs.plot_data(outdir='./outdir10', label=label)

# time delay between two detectors is
# # detector_1_vertex = IFOs[detector_1].vertex
# # detector_2_vertex = IFOs[detector_2].vertex
# # time_delay = bilby.gw.utils.time_delay_geocentric(detector_1_vertex ,detector_1_vertex , ra_vec, dec_vec, 0)
# # time_delay = bilby.gw.utils.time_delay_geocentric(position[detector_1,:],position[detector_2,:],ra_vec, dec_vec,0)
# # print('TImeDelay',time_delay)
#
# ## derivative w.r.t. theta
# d_omega_d_theta = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
# print('d_omega_d_theta is', d_omega_d_theta)
# print('shape of d_omega_d_theta is', np.shape(d_omega_d_theta))
#
# ## derivative w.r.t. phi
# d_omega_d_phi = np.array([- np.sin(theta) * np.sin(phi), np.sin(theta) * np.cos(phi), np.cos(theta)])
# print('d_omega_d_phi is', d_omega_d_phi)
# print('shape of d_omega_d_phi is', np.shape(d_omega_d_phi))
#
# ## delta_x is displacement vector between detectors i.e. Seperation vector between central station of two detector sites
# # radius_of_earth = 6371.  # radius of earth in km
# # CE_lat = 46 + 27. / 60 + 18.528 / 3600  # bilby.gw.detectora('CE.interferometer')
# # CE_long = -(119 + 24. / 60 + 27.5657 / 3600)
# # ET_lat = 40.5213
# # ET_long = 9.4248
# #
# # delta_x = radius_of_earth * acos(
# #     sin(CE_lat) * sin(ET_lat) + cos(CE_lat) * cos(ET_lat) * cos(CE_long - ET_long))
#
# # ## unit vector are
# # u1 = (position[detector_2,0] - position[detector_1,0])/delta_x
# # u2 = (position[detector_2,1] - position[detector_1, 1])/delta_x
# # u3 = (position[detector_2,2] - position[detector_1, 2])/delta_x