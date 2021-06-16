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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

current_direc = os.getcwd()
print("Current Working Directory is :", current_direc)

## Specify the output directory and the name of the simulation.
outdir = 'Overlap_reduction_function_1'
label = 'Overlap_reduction_function_1'
bilby.utils.setup_logger(outdir=outdir, label=label)

ifos = ['L1', 'H1','CE', 'ET_D_TR']
# ifos = ['CE', 'ET_D_TR']

## Set the duration and sampling frequency of the data segment that we're going to inject the signal into
start_time = 1198800017  # for one year beginning
end_time = 1230336017  # for one year long run.

## Total Duration for data taking is
duration = end_time - start_time
print('Total Duration is {}'.format(duration))

# Sampling frequency/rate, or number of measurements per second
sampling_frequency = 2048.

## Nquist Frequency i.e. maximum frequency that can be analyzed.
Nquist = sampling_frequency/2

## Divide the duration in number of segments i.e. n_seg.
#number of time segment
n_seg = 10000

## Duration of each segment is
duration_seg = duration / n_seg
print('Duration of one segment (duration_seg) is {}'.format(duration_seg))

## Converting duration of each segment into bits.
duration_seg = 2**(int(duration_seg)-1).bit_length()
print('Duration of one segment (duration_seg) in bits is {}'.format(duration_seg))

## Number of truncated time Segment are
n_seg = np.trunc(duration/duration_seg)
print('Number of truncated time segment are {}'.format(n_seg))

## Number of samples in each segment of duration (i.e. duration_seg) are (or number of FFT bins in duration_seg)
N_samples = int(duration_seg * sampling_frequency)
print('Number of Samples in each duration_seg are {}'.format(N_samples))

## sampling interval/time : time interval between successive samples
delta_T = duration_seg / N_samples
print('delta_T',delta_T)

## The frequency resolution does not depend on the length of FFT, but the length of the total sampling time T, i.e. it's 1/T,
## frequency resolution/ FFT bin resolution i.e. frequency per bin of FFT
frequency_resolution = 1 / duration_seg ## frequency_resolution in general is FR = sampling_frequency/(N_samples) = 1/duration_seg
print('frequency_resolution',frequency_resolution)

# ## FFT bins
# FFT_bins = N_samples/2
# print('FFT_bins',FFT_bins)
# FFT_bins_rsolution = Nquist/FFT_bins
# print('FFT_bins_rsolution',FFT_bins_rsolution)

## dividing the duration_seg into sub segments
n_T_seg = 512

## duration of each sub_segment is
T_seg = duration_seg / n_T_seg  # T = 8 sec
T_seg = 2 ** (int(T_seg) - 1).bit_length()
print('T_seg', T_seg)

## sampling Interval i.e. spacing between two T_seg points.
delta_T_seg = T_seg / sampling_frequency
print('delta_T_seg',delta_T_seg)

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
# print('freq_series',freq_series)
# print(len(freq_series))

## frequency bin resolution for number of bins is
n_freq_resl = Nquist/n_frequencies
print('n_freq_resl',n_freq_resl)

## fftfreq : Return the Discrete Fourier Transform sample frequencies.
#The returned float array f contains the frequency bin centers in cycles per unit of the sample spacing (with zero at the start).
#For instance, if the sample spacing is in seconds, then the frequency unit is cycles/second.
# sample spacing = d. Sample spacing is inverse of the sampling rate.

## rfftfreq : Return the real Discrete Fourier Transform sample frequencies.
freq_rfft = rfftfreq(n_samples_fft, d=1 / sampling_frequency)
# print('freq_rfft',freq_rfft)

n_freq_len = len(freq_rfft)
# print('n_freq_len',n_freq_len)

## Extracting the real part of frequency
frequency = np.append([0],freq_rfft[1::2])
print('frequency',frequency)
print(np.shape(frequency))

IFOs = bilby.gw.detector.InterferometerList(ifos)
# IFOs.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=duration_seg, start_time=0)


## Test : To understand how antenna_pattern works
# ra = 0.31597443
# dec = 0.1802837
# for ifo in IFOs:
#     # A 3x3 array representation of the detector tensor. 3x3 represent all possible combination over three unit vector.
#     detector_tensor = ifo.detector_tensor
#     print('detector tensor is', detector_tensor)
#
#     # Calculate the polarization tensor for a given sky location and time
#     # A 3x3 representation of the polarization_tensor for the specified mode.
#     polarization_tensor = bilby.gw.utils.get_polarization_tensor(ra, dec, 0, 0, 'plus') ## time = 0, phi = 0
#     print('polarization tensor is', polarization_tensor)
#
#     antenna_patteren = ifo.antenna_response(ra,dec, 0 , 0, 'plus')
#     print('antenna_patteren', antenna_patteren)

# detector_tensor = np.zeros((len(IFOs), 3,3))
# polarization_tensor = np.zeros((len(IFOs), len(ra_vec), len(ra_vec), 3,3))
# antenna_patteren = np.zeros((len(IFOs), len(ra_vec)))
# ci = 0
# for ifo in IFOs:
#     cr = 0
#     for ra in ra_vec:
#         cd = 0
#         for dec in dec_vec:
#             # A 3x3 array representation of the detector tensor. 3x3 represent all possible combination over three unit vector.
#             detector_tensor[ci ] = ifo.detector_tensor
#             # print('detector tensor is', detector_tensor)
#
#             # Calculate the polarization tensor for a given sky location and time
#             # A 3x3 representation of the polarization_tensor for the specified mode.
#             polarization_tensor[ci , cr, cd] = bilby.gw.utils.get_polarization_tensor(ra, dec, 0, 0, 'plus') ## time = 0, phi = 0
#             print('polarization tensor is', polarization_tensor)
#
#             antenna_patteren = ifo.antenna_response(ra,dec, 0 , 0, 'plus')
#             # print('antenna_patterne', antenna_patteren)
#             cd += 1
#         cr += 1
#     ci += 1
# print(detector_tensor)
# print(polarization_tensor)
# print(antenna_patteren)


#########################################################
''' Overlap Reduction Function Calculation : Approach-1 '''
#########################################################

modes = ['plus', 'cross']  # ,'breathing']

## Calculate the position of the detector vertex in geocentric coordinates in meters.
position = np.zeros((len(IFOs), 3))
cp = 0
for ifo in IFOs:
    position[cp, :] = ifo.vertex_position_geocentric()
    cp += 1
# print(' position of the detector vertex',position)
# print('shape of position of the detector vertex', np.shape(position))

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

##  np.meshgrid(x, y)
# x = one dim np.array of length m
# y = one dim np.array of length n
## if m > n i.e. m != n
# x = ([-2,-1,0,1,2]) m =5
# y = ([-2,-1,0,1]) n = 4
# total number of 20 coordinates in xy plane
# [xx, yy] = np.meshgrid(x,y)
# xx = horizontal stack of x np.array upto m*n , m = no of columns, n = no of columns
# yy = transpose and verticle stack of y np.array upto m*n .
## np.flatten flat a matrix of ndim array to  one dim array
## e.g. x = ([1,2,3], [4,5,6])
# x.flatten() = ([1,2,3,4,5,6])
## y = ([[1,2,3], [4,5,6]], [[1,2,3], [4,5,6]]) ## two dim array
# y.flatten() =  ([1,2,3,4,5,6], [1,2,3,4,5,6])

[ra_mat, dec_mat] = np.meshgrid(ra_vec, dec_vec)
ra_vec = ra_mat.flatten()
dec_vec = dec_mat.flatten()

# ra_vec.resize(frequency.shape)
# dec_vec.resize(frequency.shape)

''' 
F_+ and F_× will depend on the three angles defining the sky position and polarization basis.
using equatorial co¨ordinates, they will depend on right ascension, declination and the polarization angle ψ.
Note antenna_response in bilby.gw.antenna_response gives  dot product between detector_tensor and polarization tensor.
'''

# delta_x = position[0, :] - position[1, :]
# omega = np.array([np.sin(dec_vec) * np.cos(ra_vec), np.sin(dec_vec) * np.sin(ra_vec), np.cos(dec_vec)])
#
# detector_tensor = np.zeros((len(IFOs),3,3))
# ci = 0
# for ifo in IFOs:
#     detector_tensor[ci,:] = ifo.detector_tensor
#     ci += 1
# # print('detector_tensor is', detector_tensor)
#
# overall_detector_tensor = np.matmul(detector_tensor[0,:,:], detector_tensor[1,:,:])
# # print('overall_detector_tensor is', overall_detector_tensor)
#
# polarization_tensor = np.zeros((len(IFOs),len(modes), len(ra_vec), 3, 3))
# ci = 0
# for ifo in IFOs:
#     cm = 0
#     for mode in modes:
#         idx = 0
#         for idx in range(len(ra_vec)):
#             polarization_tensor[ci,cm,idx,] = bilby.gw.utils.get_polarization_tensor(ra_vec[idx], dec_vec[idx], 0, 0, mode) # time = 0 , psi = 0.
#             idx +=1
#         cm += 1
#     ci += 1
# # print('polarization_tensor  is ',polarization_tensor )
# # print(np.shape(polarization_tensor ))

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
# print('antenna_response',antenna_response)
# print('shape of antenna_patteren', np.shape(antenna_response))

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
            # print('plus mode', plus_mode)
            cross_mode = np.einsum('i,i->i',f1c, f2c)
            # print('cross mode',cross_mode)

            total_response = np.einsum('i,i->i',plus_mode, cross_mode)
            # print('total_response',total_response)

            # combined_antenna_response = np.dot(f1p , f2p) + np.dot(f1c , f2c)
            # print('combined_antenna_response', combined_antenna_response)

            delta_x = position[detector_1, :] - position[detector_2, :]
            omega = np.array([np.cos(dec_vec) * np.cos(ra_vec), np.cos(dec_vec) * np.sin(ra_vec), np.sin(dec_vec)])
            ## The unit vector Ω is a direction on the 2-D sphere (sky), described by two angles (θ, ϕ),
            ## from which the GW is arriving. We can therefore write the wavevector as k = 2πf Ω/c

            orf[di,:] = (5 / (8 * np.pi)) * np.sum(np.outer(eins, np.cos(dec_vec) * (f1p * f2p + f1c * f2c)) * np.exp(
                        1j * 2 * np.pi * np.outer(frequency, np.dot(omega.T, delta_x)) / speed_of_light), axis=1) * d_dec * d_ra

            di += 1

    return orf

gamma = odf(frequency)
# print('gamma',gamma)
# print(np.shape(gamma))

# ## Use this for loop for a pair of four detectors.
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
#     plt.plot(frequency, gamma[idx, :], label = label)
#     legend = plt.legend(loc='lower right', fontsize='small')
#     # plt.xscale('log')
#     # plt.xlim(1, 1000)
#     plt.xlim(0, 600)
#     # plt.axhline(y=0)
#     # plt.axvline(x=0)
#     plt.xlabel(r'f (Hz)')
#     plt.ylabel(r'gamma(f)')
#     # plt.title(r'Overlap Reduction Function Between LIVINGSTON and HANFORD')
#     plt.title(r' ORF Between' + label)
#     plt.savefig('./Overlap_reduction_function/ORF ' + label)
#     # plt.savefig('./Overlap_reduction_function/ORF_log ' + label)
#     plt.show()
#     plt.close()

# plt.figure(16)
# plt.plot(frequency, gamma[9, :], label = 'CE & ET_D_TR_1')
# plt.plot(frequency, gamma[10, :], label = 'CE & ET_D_TR_2')
# plt.plot(frequency, gamma[11, :], label = 'CE & ET_D_TR_3')
# # plt.plot(frequency, gamma[12, :], label = 'ET_D_TR_1 & ET_D_TR_2')
# # plt.plot(frequency, gamma[13, :], label = 'ET_D_TR_1 & ET_D_TR_3')
# # plt.plot(frequency, gamma[14, :], label = 'ET_D_TR_2 & ET_D_TR_3')
# legend = plt.legend(loc='lower right', fontsize='small')
# # plt.xscale('log')
# # plt.xlim(1, 1000)
# plt.xlim(0, 600)
# # plt.axhline(y=0)
# # plt.axvline(x=0)
# plt.xlabel(r'f (Hz)')
# plt.ylabel(r'gamma(f)')
# plt.title(r'Overlap Reduction Function Between CE and ET_D_TR')
# plt.savefig('./Overlap_reduction_function/Overlap Reduction Function')
# # plt.savefig('./Overlap_reduction_function/Overlap Reduction Function_log')
# plt.show()
# plt.close()

#########################################################
''' Overlap Reduction Function Calculation : Approach-2 '''
#########################################################

def gamma(frequency):

    ## Eq. (3.31) of Allen and Romano Paper
    detector_tensor = np.zeros((len(IFOs),3,3))
    ci = 0
    for ifo in IFOs:
        detector_tensor[ci,:] = ifo.detector_tensor
        ci += 1
    print('detector_tensor is', detector_tensor)
    print(np.shape(detector_tensor))

    gamma = np.zeros((int(len(IFOs) * (len(IFOs) - 1) / 2), len(frequency)))
    di = 0
    ## Iterate Over first detectors
    for detector_1 in range(len(IFOs)):

        p1_0 = position[detector_1, 0]
        p1_1 = position[detector_1, 1]
        p1_2 = position[detector_1, 2]

        detector_tensor_1 = detector_tensor[detector_1,:]

        ## Iterate Over second detectors
        for detector_2 in range(detector_1 + 1, len(IFOs)):

            p2_0 = position[detector_2, 0]
            p2_1 = position[detector_2, 1]
            p2_2 = position[detector_2, 2]

            detector_tensor_2 = detector_tensor[detector_2, :]

            ## computing distance vector between two detectors
            s0 =  p1_0 - p2_0
            s1 =  p1_1 - p2_1
            s2 =  p1_2 - p2_2
            d = np.sqrt(s0 ** 2 + s1 ** 2 + s2 ** 2)

            ## Eq. (3.32) of Allen and Romano (1999)
            alpha = 2.0 * np.pi * frequency * d / speed_of_light

            ## Change s into a unit vector points in the direction connecting two detectors.
            s0 /= d
            s1 /= d
            s2 /= d

            def s(j):

                s0 = p1_0 - p2_0
                s1 = p1_1 - p2_1
                s2 = p1_2 - p2_2
                d = np.sqrt(s0 ** 2 + s1 ** 2 + s2 ** 2)

                if j == 0:
                    s0 = s0 / d
                    return s0
                if j == 1:
                    s1 = s1 / d
                    return s1
                if j == 2:
                    s2 = s2 / d
                    return s2

            ## Eq. (3.40) of Allen and Romano (1999)
            ## Standard spherical bessel function
            j0 = np.divide(np.sin(alpha) ,alpha)
            j0[np.isnan(j0)] = 0

            j1 = np.divide(np.sin(alpha) , alpha ** 2) - np.divide(np.cos(alpha), alpha)
            j1[np.isnan(j1)] = 0

            j2 = 3. * np.divide(np.sin(alpha), alpha ** 3) - 3. * np.divide(np.cos(alpha), alpha ** 2) - np.divide(np.sin(alpha), alpha)
            j2[np.isnan(j2)] = 0

            ## Eq. (3.44) of Allen and Romano (1999)
            rho1 = 0.5 * np.divide(( 10.0 * alpha ** 2 * j0 - 20.0 * alpha * j1 +  10.0 * j2), alpha ** 2)
            rho1[np.isnan(rho1)] = 0

            rho2 = 0.5 * np.divide((-20.0 * alpha ** 2 * j0 + 80.0 * alpha * j1 - 100.0 * j2), alpha ** 2)
            rho2[np.isnan(rho2)] = 0

            rho3 = 0.5 *np.divide(( 5.0  * alpha ** 2 * j0 - 50.0 * alpha * j1 + 175.0 * j2), alpha ** 2)
            rho3[np.isnan(rho3)] = 0

            ## The detector_tensor is given by d^{ab} = 1/2 (X_{i}^{a} X_{i}^{b} − Y_{i}^{a} Y_{i}^{b})
            # The tensor d depends on the location and orientation of the detector on the Earth, and also on the sidereal time
            # tensors e^+ and e^× depend on the location of the source on the sky and also on the polarization angle ψ
            # For a given detector at a given time, F+ and F× thus depend on the source’s sky position (α, δ) and ψ
            # i represent the number of detector

            # ## Eq. (3.31) of Allen and Romano Paper
            # d_t = np.zeros((len(IFOs), 3, 3));
            # ci = 0
            # for ifo in IFOs:
            #     d_t[ci,:] =  ifo.detector_tensor
            #     ci += 1

            d1d2   = 0
            sd1d2s = 0
            sd1s1  = 0
            sd2s2  = 0
            for i in range(0,3):
                sd1 = 0
                sd2 = 0
                for j in range(0, 3):
                    d1d2 += detector_tensor_1[i,j] * detector_tensor_2[i,j]
                    sd1 += detector_tensor_1[i,j] * s(j)
                    sd2 += detector_tensor_2[i,j] * s(j)
                sd1s1 += sd1 * s(i)
                sd2s2 += sd2 * s(i)
                sd1d2s += sd1 * sd2

            # ## Eq (3.43) of Allen and Romano (1999)
            gamma[di,:] = rho1 * d1d2 + rho2 * sd1d2s + rho3 * sd1s1 * sd2s2
            # print(np.isnan(gamma).any())
            # print(np.isinf(gamma).any())

            di += 1

    return gamma

# # frequency = (freq_rfft)
# gamma_1 = gamma(frequency)
# print('gamma',gamma_1)
# print(np.shape(gamma_1))
#
# ## Use this for loop for a pair of four detectors.
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
#     plt.plot(frequency, gamma_1[idx, :], label = label)
#     legend = plt.legend(loc='lower right', fontsize='small')
#     plt.xscale('log')
#     plt.xlim(1, 1000)
#     # plt.xlim(0, 600)
#     # plt.axhline(y=0)
#     # plt.axvline(x=0)
#     plt.xlabel(r'f (Hz)')
#     plt.ylabel(r'gamma(f)')
#     # plt.title(r'Overlap Reduction Function Between LIVINGSTON and HANFORD')
#     plt.title(r' ORF Between' + label)
#     # plt.savefig('./Overlap_reduction_function_1/ORF ' + label)
#     plt.savefig('./Overlap_reduction_function/ORF_II_log ' + label)
#     plt.show()
#     plt.close()

# plt.figure(16)
# plt.plot(frequency, gamma_1[9, :], label = 'CE & ET_D_TR_1')
# plt.plot(frequency, gamma_1[10, :], label = 'CE & ET_D_TR_2')
# plt.plot(frequency, gamma_1[11, :], label = 'CE & ET_D_TR_3')
# # plt.plot(frequency, gamma_1[12, :], label = 'ET_D_TR_1 & ET_D_TR_2')
# # plt.plot(frequency, gamma_1[13, :], label = 'ET_D_TR_1 & ET_D_TR_3')
# # plt.plot(frequency, gamma_1[14, :], label = 'ET_D_TR_2 & ET_D_TR_3')
# legend = plt.legend(loc='lower right', fontsize='small')
# plt.xscale('log')
# plt.xlim(1, 1000)
# # plt.xlim(0, 600)
# # plt.axhline(y=0)
# # plt.axvline(x=0)
# plt.xlabel(r'f (Hz)')
# plt.ylabel(r'gamma(f)')
# plt.title(r'Overlap Reduction Function Between CE and ET_D_TR')
# # plt.savefig('./Overlap_reduction_function_1/Overlap Reduction Function')
# plt.savefig('./Overlap_reduction_function/Overlap Reduction Function_log_II')
# plt.show()
# plt.close()

