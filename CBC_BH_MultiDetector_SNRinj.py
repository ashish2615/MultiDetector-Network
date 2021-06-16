from __future__ import division, print_function

import numpy as np
from numpy import loadtxt

import sys
import bilby
from bilby import prior
import logging
import deepdish
import pandas as pd
import h5py

import seaborn as sns
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

import platform

# print(platform.python_version())   # This command will check which version of python you are using in virtual environment.
# print(platform.sys.version)
# Use this command to check any arthematic errors and warnings in the system.
# np.seterr(all='raise')

idx = 1 #int(sys.argv[1]) ## number of injection
#ifos = ['CE']
ifos = ['CE','ET_D_TR']  ## sys.argv[2].split(',')
sampler = 'dynesty'       ## sys.argv[3]

# Specify the output storage directory and the name of simulation identifier to apply to output files..
outdir = 'outdir0'
label = 'sample_param_' + str(idx)
extension='json'
bilby.utils.setup_logger(outdir=outdir, label=label)

## Set the duration and sampling frequency of the data segment that we're going to inject the signal into
duration = 4.                     # time duration in sec.
sampling_frequency = 2048.        # Sampling Frequency in Hz

## Loading the injection parameters from hdf5 file
injection_parameters = dict(deepdish.io.load('injections_test.hdf5')['injections'].loc[idx])

## Changing 'iota' to 'theta_jn' to be suitable with bilby
injection_parameters['theta_jn'] = injection_parameters.pop('iota')

## we use start_time=start_time to match the start time and wave interferome time. if we do not this then it will create mismatch between time.
start_time = injection_parameters['geocent_time']
#print("injection time = {}".format(start_time))

## Redshift to luminosity Distance conversion using bilby
injection_parameters['luminosity_distance'] = bilby.gw.conversion.redshift_to_luminosity_distance(injection_parameters['redshift'])

# First mass needs to be larger than second mass
if injection_parameters['mass_1'] < injection_parameters['mass_2']:
    tmp = injection_parameters['mass_1']
    injection_parameters['mass_1'] = injection_parameters['mass_2']
    injection_parameters['mass_2'] = tmp
print(injection_parameters)

# Fixed arguments passed into the source model : A dictionary of fixed keyword arguments
# to pass to either `frequency_domain_source_model` or `time_domain_source_model`.
waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
            reference_frequency=50., minimum_frequency=2.)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = bilby.gw.WaveformGenerator(
      duration=duration, sampling_frequency=sampling_frequency, start_time=start_time,
      frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
      waveform_arguments=waveform_arguments)

# create the frequency domain signal
#hf_signal = waveform_generator.frequency_domain_strain()

## Initialization of GW interferometer
IFOs = bilby.gw.detector.InterferometerList(ifos)

# Generates an Interferometer with a power spectral density based on advanced LIGO.
IFOs.set_strain_data_from_power_spectral_densities(
         sampling_frequency=sampling_frequency, duration=duration,
         start_time=start_time #start_time=injection_parameters['geocent_time'] - duration/2)
          )
# Above line we use start_time=start_time to match the start time and wave interferometer time scale.
# one can also set  both time scale to : start_time=injection_parameters['geocent_time'] - duration/2.

IFOs.inject_signal(waveform_generator=waveform_generator,
               parameters=injection_parameters)

# print("start_time = {}".format(getattr(waveform_generator, 'start_time')))

mf_snr = np.zeros((1, len(IFOs)))[0]
waveform_polarizations = waveform_generator.frequency_domain_strain(injection_parameters)
k = 0
for ifo in IFOs:
    signal_ifo = ifo.get_detector_response(waveform_polarizations, injection_parameters)
    mf_snr[k] = ifo.matched_filter_snr(signal=signal_ifo)
    if np.isnan(mf_snr[k]):
        mf_snr[k] = 0.
    print('{}: SNR = {:02.2f} at z = {:02.2f}'.format(ifo.name, mf_snr[k], injection_parameters['redshift']))
    k += 1

# print(injection_parameters['redshift'])
#  print(np.asarray(mf_snr))
#  test = np.column_stack(([injection_parameters['redshift']], np.asarray(mf_snr)))
#  print(test)
#  df = pd.DataFrame(test)
#  df.to_csv("file_path.csv", header=None)
#  np.savetxt("output.csv", np.column_stack((injection_parameters['redshift'], np.asarray(mf_snr))), delimiter=",")

# injection_parameters.pop('redshift')
# injection_parameters.pop('luminosity_distance')
# print(injection_parameters)

priors = bilby.gw.prior.BBHPriorDict(filename='binary_black_holes_cosmo_uniform.prior') # binary_black_holes_cosmo.prior

pzDlBH = np.loadtxt("PzDlBH.txt", delimiter=',')

priors['geocent_time'] = bilby.core.prior.Uniform(
           minimum=injection_parameters['geocent_time'] - 1,
           maximum=injection_parameters['geocent_time'] + duration,
           name='geocent_time', latex_label='$t_c$', unit='$s$')

priors['luminosity_distance'] = bilby.core.prior.Interped(
     name='luminosity_distance', xx=pzDlBH[:, 1], yy=pzDlBH[:, 2], minimum=1e1, maximum=1e4, unit='Mpc')

#priors.pop('redshift')
#priors.pop('luminosity_distance')
#print(priors)

#priors['luminosity_distance'] = bilby.core.prior.Uniform(
 #       name='luminosity_distance', minimum=1e2, maximum=1e4, unit='Mpc')  # Here change the max value of Luminosity Distnace

## For parameters vary only in mass_1 and mass_2 and rest of all are fixed : use this one
#priors['mass_1'] = bilby.core.prior.PowerLaw(
#        name='mass_1', alpha=-1, minimum=5, maximum=50, unit='$M_{\\odot}$')  ## One can also use bilby.core.prior.Uniform depending on their own interest.
#priors['mass_2'] = bilby.core.prior.PowerLaw(
#        name='mass_2', alpha=-1, minimum=5, maximum=50, unit='$M_{\\odot}$')


## list of injection_parameters key that are not being sampled
for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'psi', 'ra',
                 'dec', 'geocent_time', 'phase']:
    #print(key)
    priors[key] = injection_parameters[key]
del priors['redshift']
priors['theta_jn'] = priors.pop('iota')
#print('Re-defined priors are {}'.format(priors))
logging.info(priors.keys())

# Initialise the likelihood by passing in the interferometer data (IFOs) and the waveform generator
likelihood = bilby.gw.GravitationalWaveTransient(interferometers=IFOs, waveform_generator=waveform_generator,
                                      time_marginalization=False, phase_marginalization=False,
                                      distance_marginalization=False, priors=priors)

sampling_seed = np.random.randint(1, 1e6)
np.random.seed(sampling_seed)
logging.info('Sampling seed is {}'.format(sampling_seed))

sampler_dict = dict()
sampler_dict['dynesty'] = dict(npoints=1000)
sampler_dict['pymultinest'] = dict(npoints=1000, resume=False)
sampler_dict['emcee'] = {'nwalkers': 40, 'nsteps': 20000, 'nburn': 2000}

result = bilby.run_sampler(
        likelihood=likelihood, priors=priors, sampler='dynesty', npoints=1000,
        injection_parameters=injection_parameters, outdir=outdir, label=label)

# result = bilby.run_sampler(likelihood=likelihood, priors=priors, sampler=sampler,
#                            injection_parameters=injection_parameters, outdir=outdir,
#                            label=label, **sampler_dict[sampler])
# make some plots of the outputs
#print(result)

result.plot_corner()
plt.close()


#############################
## dictionary to text
#############################

## Create a injection parameters file with txt format.
#injection_file = open('./outdir/injection_parameters_'+str(idx)+'.txt', mode = 'w')
#print(injection_file)
## saving data file to txt format
#injection_file_save = injection_file.write(str(injection_parameters))  ## this worked
#injection_file.close()


###########################
# dictionary to numpy array
###########################
## define a numpy array of data (as injection_paramteres are in dictonary)
injection_parameters_array = np.array(injection_parameters.values())
#print(injection_parameters_array)

## Create a injection parameters file with txt format.
injection_file = open('./outdir0/injection_parameters_' + str(idx) +'.txt', mode = 'w')
#print(injection_file)

## saving data file to txt format
injection_file_save = injection_file.write(str(injection_parameters_array))
injection_file.close()

injection_parameters_save = np.save('./outdir0/injection_parameters_'+ str(idx) +'.npy',injection_parameters)
#injection_parameters_load = np.load('./outdir/injection_parameters.npy')
#print(injection_parameters_load)
#print(injection_parameters_load.item())
#print(injection_parameters_load.item().keys())
#print(injection_parameters_load.item().values())

'''
injection_parameters = dict(
mass_1    = source frame (non-redshifted) primary mass (solar masses)
mass_2    = source frame (non-redshifted) secondary mass (solar masses)
a_1       = primary dimensionless spin magnitude
a_2       = secondary dimensionless spin magnitude
tilt_1    = polar angle between primary spin and the orbital angular momentum (radians)
tilt_2    = polar angle between secondary spin and the orbital angular momentum 
phi_12    = azimuthal angle between primary and secondary spin (radians)
phi_jl    = azimuthal angle between total angular momentum and orbital angular momentum (radians)
iota      = inclination angle between line of sight and orbital angular momentum (radians)
phase     = phase (radians)
ra        = source right ascension (radians)
dec       = source declination (radians)
psi       =  gravitational wave polarisation angle
luminosity_distance  = luminosity distance to source (Mpc)
waveform_approximant ='IMRPhenomPv2', # waveform approximant name
reference_frequency  = gravitational waveform reference frequency (Hz)
geocent_time   = reference time at geocentre (time of coalescence or peak amplitude) (GPS seconds)
)
'''
