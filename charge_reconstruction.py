import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

from ctapipe.io import EventSource
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.io import TableLoader
import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz
from ctapipe.containers import ImageParametersContainer
from ctapipe.calib import CameraCalibrator
from ctapipe.image import ImageProcessor
from ctapipe.reco import ShowerProcessor
from ctapipe.io import DataWriter
from copy import deepcopy, copy
import tempfile
from ctapipe.visualization import CameraDisplay
from ctapipe.image import timing_parameters
from abc import ABC, abstractmethod

from ctapipe.instrument import SubarrayDescription, TelescopeDescription, CameraDescription, CameraGeometry
from ctapipe.containers import ArrayEventContainer
from astropy.time import Time
from scipy import stats
from ctapipe.image import ImageModifier
from ctapipe.io import SimTelEventSource

import ctapipe_process as ctapc
from ctapipe_process import compute_h5

input_url = "Data/gamma_20deg_0deg_run1555___cta-prod6-paranal-2147m-Paranal-dark-bs5.0-10k-lfa64.simtel.zst"   #Input simtel file
source = EventSource(input_url)
subarray = source.subarray

# ctapipe Components that are configurable per telescope (type)
# need the subarray information
calib = CameraCalibrator(source.subarray)


for event in source:
    calib(event)
    break


#Initialize arrays
true_pe = np.array([])
recon_pe = np.array([])
noise_pe = np.array([])
total_pe = np.array([])


#Disabled pixels
tel_id = next(iter(event.dl1.tel.keys()))
flashcam_disabled_pixels = event.mon.tel[tel_id].pixel_status.hardware_failing_pixels[0]

# Good pixels
godd_pixels = ~flashcam_disabled_pixels


# Load corresponding tables

file = compute_h5(input_file = input_url, extractor = "NeighborPeakWindowSum", width = 7, shift = 2)  #, std=2)  # Generate h5 and use it
print(file)

loader = TableLoader(file, 
                     load_true_images = True, 
                     load_dl1_images = True,
                     load_dl1_parameters=False,
                    )

events = loader.read_telescope_events_by_type()



for tel_type, table in events.items():
    if tel_type == 'MST_MST_FlashCam':
            good_pixels = ~flashcam_disabled_pixels
    else:
        good_pixels = slice(None)

    images = table["image"]
    true_images = table["true_image"]
    pixel_values = images[:, good_pixels]
    true_pixel_values = true_images[:, good_pixels]

    noise = pixel_values[true_pixel_values == 0]

    if tel_type == 'MST_MST_FlashCam':
        true_pe = np.append(true_pe, true_pixel_values[true_pixel_values > 0])
        recon_pe = np.append(recon_pe, pixel_values[true_pixel_values > 0])
        noise_pe = np.append(noise_pe, noise)
        total_pe = np.append(total_pe, pixel_values)


def compute_profile(x, y, nbin=(100,100)):
    """
    Compute profile plots
    """

    h, xe, ye = np.histogram2d(x,y,nbin)
    
    # bin width
    xbinw = xe[1]-xe[0]

    # getting the mean and RMS values of each vertical slice of the 2D distribution
    x_array      = []
    x_slice_mean = []
    x_slice_rms  = []
    for i in range(xe.size-1):
        yvals = y[ (x > xe[i]) & (x <= xe[i+1]) ]
        if yvals.size > 0:
            x_array.append(xe[i] + xbinw/2)
            x_slice_mean.append(yvals.mean())
            x_slice_rms.append(yvals.std())
    x_array = np.array(x_array)
    x_slice_mean = np.array(x_slice_mean)
    x_slice_rms = np.array(x_slice_rms)

    return x_array, x_slice_mean, x_slice_rms


# SIGNAL, NOISE DISTRIBUTIONS


QUANTILES = [0.9, 0.99, 0.999]

bins = np.logspace(-1.0, 4.0, 101)
hist_kwargs = dict(
    bins=bins,
    histtype='step',
    cumulative=-1,
)

fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(8, 6))

plt.hist(true_pe, **hist_kwargs, label='Signal-Only Pixels', linewidth=1.5)
plt.hist(noise_pe, **hist_kwargs, label='Noise-Only Pixels', linewidth=1.5)

quantiles = np.quantile(noise_pe, QUANTILES)
for v, q, alpha in zip(quantiles, QUANTILES, np.linspace(0.3, 0.8, len(QUANTILES))):
    plt.axvline(v, label=f'Q({q:.1%}) = {v:.2f}', color='black', alpha=alpha)

plt.legend(fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.title(r"Charge distribution", fontsize=16)
plt.xlabel('Charge', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.show()


# PROFILE PLOTS

bins = np.unique(np.round(np.logspace(0, 5, 51))) - 0.5   #Logarithmic binning

# Variable chosen to plot

x = true_pe  
y = recon_pe

p_x, p_mean, p_rms = compute_profile(x, (x-y)**2, (bins, bins))

plt.figure(figsize=(10, 8))
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title(r"Extractor: FlashCamExtractor", fontsize=14)
plt.xlabel("True charge", fontsize=14)
plt.ylabel("Reconstructed charge", fontsize=14)
#plt.errorbar(p_x, p_mean, p_rms, fmt='_', ecolor='b', color='b', label='Observed')
plt.scatter(p_x, np.sqrt(p_mean)/p_x, color='black')
plt.plot(p_x, 1/np.sqrt(p_x), color='black', label='Expected')
plt.legend(fontsize=14)
plt.show()
