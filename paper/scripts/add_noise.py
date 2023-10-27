#!/usr/bin/env python3

"""
Add noise to the quasar spectra

To generate the renormalized, noisy, dataset:
python add_noise --seed 100921 -o qsos_noisy.fits --renorm --add_noise
"""

import ast
import time

import numpy as np
import matplotlib.pyplot as plt
import fitsio
from astropy.table import Table

# These are necessary to generate the scaling quasar
from astropy.cosmology import Planck13
from simqso.sqgrids import *
from simqso import sqbase
from simqso.sqrun import buildSpectraBulk,buildQsoSpectrum
from simqso.sqmodels import BOSS_DR9_PLEpivot,get_BossDr9_model_vars

# Used in rebinning
from scipy.ndimage import binary_erosion

def get_wave(wavemin=3600, wavemax=10000, dloglam=1e-4):
    """
    Return logarithmic wavelength array from wavemin to wavemax step dloglam

    Args:
        wavemin: minimum wavelength
        wavemax: maximum wavelength
        dloglam: stepsize in log(wave)

    Return: wave array
    """
    n = np.log10(wavemax/wavemin) / dloglam
    wave = 10**(np.log10(wavemin) + dloglam*np.arange(n))
    return wave

def deshift_and_stack(fl, redshifts, w_obs):
    X = []
    waves = []
        
    for i in range(fl.shape[0]):
        z = redshifts[i]
        shifted_wave = w_obs / (1 + z)
        
        # Normalize to "approximately median of 1 in erg space" then
        # convert to photon space for poisson sampling
        X.append(fl[i] / np.median(fl[i]) * shifted_wave)
        waves.append(shifted_wave)
        
    return X, waves


def generate_scaling_qso(wave, seed=100921, z_gen=2.1):
    # Generates a single QSO with the same parameters but covering
    # the entire covered wavelength range for normalization purposes
    kcorr = sqbase.ContinuumKCorr('DECam-r',1450,effWaveBand='SDSS-r')
    qsos = generateQlfPoints(BOSS_DR9_PLEpivot(cosmo=Planck13),
                             (16, 16), (2, 3),
                             kcorr=kcorr, zin=[z_gen],
                             qlfseed=seed, gridseed=seed)

    sedVars = get_BossDr9_model_vars(qsos, wave, noforest=True)
    qsos.addVars(sedVars)
    qsos.loadPhotoMap([('DECam','DECaLS'),('WISE','AllWISE')])

    _, spectra_for_norm = buildSpectraBulk(wave * (1 + z_gen), qsos, saveSpectra=True, maxIter=5, verbose=10)
    spectra_for_norm = spectra_for_norm[0]
    # Normalize to "approximately median of 1 in erg space"
    # Since we'll do the normalization to the template in erg space we don't need
    # to convert to photon space
    return spectra_for_norm / np.median(spectra_for_norm)

def rebin_to_common(w_in, w_out, fl, iv, fl_idx):
    # This spectra wasn't contained in cutout area
    if len(w_in) < 1:
        return None, None
    
    l_min = w_out[0]
    l_max = w_out[-1]
    dl = w_out[1] - w_out[0]
    
    # We rebin by considering how much (percent) of flux
    # should go to the lower bin and how much to the higher bin
    # Since both grids are on the exact same spacing
    # this is just a matter of finding how much the two
    # bins overlap each other by.
    idx_in = np.argmax(w_in > l_min)
    idx_out = np.argmax(w_out > w_in[idx_in])
    d_low = w_out[idx_out] - w_in[idx_in]
    d_high = w_in[idx_in + 1] - w_out[idx_out]
    
    # Gets the bin in the output grid that includes the lower edge 
    # of the input grid. Can add one for upper edge since
    # both grids have the same spacing.
    bin_low = np.floor((w_in - l_min) / dl).astype(int)
    good_low = (bin_low >= 0) & (bin_low < nbins_final)
    bin_high = bin_low + 1
    good_high = (bin_high >= 0) & (bin_high < nbins_final)
    
    # Not enough of this grid actually falls into the common grid
    if not np.any(good_low) or not np.any(good_high):
        return None, None

    # If the grid exactly aligns with the bottom of the common grid
    if np.min(bin_low[good_low]) == 0:
        base = w_out[0]
        idx_in = np.argmax(w_in[1:] > l_min)
        idx_out = np.argmax(w_out > w_in[idx_in]) - 1
        
        # Need to handle this case because subtracting 1
        # will roll it over to the end of the array
        if idx_in == -1: 
            idx_in += 1
            idx_out += 1
            base = w_out[idx_out]
            
        d_low = base - w_in[idx_in]
        d_high = w_in[idx_in + 1] - base
        
    # If the grid overlaps the edge of the upper end of the grid
    elif np.max(bin_high[good_high]) == (len(w_out) - 1):
        idx_in = np.argmax(w_in > w_out[-1]) - 1
        idx_out = -1 #np.argmax(w_out > w_in[idx_in]) - 1
        d_low = w_out[idx_out] - w_in[idx_in]
        d_high = w_in[idx_in + 1] - w_out[idx_out]
        
    percent_low = d_low / dl
    percent_high = d_high / dl

    # Inverting the inverse variance to get the variance
    var = iv
    nz = iv != 0
    var[nz] = 1 / var[nz]

    c_low = np.bincount(bin_low[good_low], weights=fl[good_low] * percent_low, minlength=len(w_out))
    c_high = np.bincount(bin_high[good_high], weights=fl[good_high] * percent_high, minlength=len(w_out))
    
    var_low = np.bincount(bin_low[good_low], weights=var[good_low] * percent_low ** 2, minlength=len(w_out))
    var_high = np.bincount(bin_high[good_high], weights=var[good_high] * percent_high ** 2, minlength=len(w_out))
    
    fl_out = c_low + c_high
    # For masking out pixels that aren't in the grid
    mask = np.zeros_like(w_out)
    mask[fl_out != 0] = 1
    
    # The binary erosion will just expand the zeros by 
    # one eliminating the bordering 1. This avoids an off by
    # one error in the mask compared to which pixels actually
    # have data
    mask = binary_erosion(mask, border_value=1)
    
    # Inverting the variance back to inverse variance
    var_out = var_low + var_high
    iv_out = var_out
    nz = np.abs(var_out) > 1e-2 # Mitigating small factor errors
    iv_out[nz] = 1 / var_out[nz]

    return fl_out, iv_out * mask, mask


def add_noise(fl, seed, add_noise=True):
    rng = np.random.default_rng(seed)
    fl_noisy = []
    iv_noisy = []
    
    poisson = add_noise
    # gaussian = add_noise
    
    for i in range(len(fl_shift)):
        # # Pick an SNR between 0.5 and 1.5 so the mean SNR is vaguely 1
        # signal = np.mean(fl_shift[i] ** 2)
        # snr_choice = rng.uniform(0.5, 1.5)
        # noise_sigma = np.sqrt(signal) / snr_choice
        # noise = rng.normal(0, noise_sigma, fl_shift[i].shape)

        if poisson:
            x_poisson = rng.poisson(fl_shift[i].clip(min=0))
        else:
            x_poisson = fl_shift[i]

        # Estimating the variance from the poisson sim
        # var_poisson = x_poisson / w_shift[i]
        # var_noise = (noise_sigma / w_shift[i]) ** 2
        
        # Pick an SNR between 0.5 and 1.5 so the mean SNR is vaguely 1
        x_poisson = x_poisson / w_shift[i]
        var_poisson = x_poisson
        
        # Pick an SNR between 0.5 and 1.5 so the mean SNR is vaguely 1
        signal = np.mean(x_poisson ** 2)
        snr_choice = rng.uniform(0.5, 1.5)
        noise_sigma = np.sqrt(signal) / snr_choice
        noise = rng.normal(0, noise_sigma, fl_shift[i].shape)
        var_noise = noise_sigma ** 2

        # Remember these are inverse variances so w_shift goes on top for
        # conversion to ergspace
        if add_noise:
            fl_sim = x_poisson + noise
            iv_noisy.append(1 / (var_poisson + var_noise))
        else:
            fl_sim = fl_shift[i] / w_shift[i]
            iv_noisy.append((fl_sim != 0))

        fl_noisy.append(fl_sim)
        
    return fl_noisy, iv_noisy


def rebin_all(fl, iv, waves, w_rest, w_scale, spectra_for_norm, renorm=True, add_noise=True):
    fl_rebinned = []
    iv_rebinned = []
    masks = []
    print(len(w_rest))
    for i in range(len(fl)):
        fl_1, iv_1, m = rebin_to_common(np.log10(waves[i]), np.log10(w_rest), fl[i], iv[i], i)
        if fl_1 is not None:
            if np.all(fl_1 < 0): print(i)
            
            if renorm:
                # Making sure to renorm the median of the spectra
                # to only the median of that region covered by
                # the global spectra
                norm_min = np.argmax(w_scale > waves[i][0])
                norm_max = np.argmax(w_scale > waves[i][-1])

                norm_val = np.median(spectra_for_norm[norm_min:norm_max])
                spec_val = np.median(fl_1[fl_1 != 0])

                scale = norm_val / np.abs(spec_val)
                normed = fl_1 * scale
                normed_iv = iv_1 / (scale ** 2)

            else:
                normed = fl_1
                normed_iv = iv_1
            
            fl_rebinned.append(normed)
            masks.append(m)
            if add_noise:
                iv_rebinned.append(normed_iv)
            else:
                iv_rebinned.append(m.astype(float))
                
    return fl_rebinned, iv_rebinned, masks
  
#-------------------------------------------------------------------------

import argparse
p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, default=1234, help="Random seed")
p.add_argument('-o', '--out', type=str, help="Output filename")
p.add_argument('--renorm', action="store_true", help="Whether to renormalize or not.")
p.add_argument('--add_noise', action="store_true", help="Whether to add noise or not.")

args = p.parse_args()

print(args.add_noise)
print(args.renorm)

# All sorts of settable hyperparameters
seed = args.seed

z_min = 0
z_max = 4

# Full range to fit for qsos from z=0 to z=4
# that overlap the eboss grid
eboss_min, eboss_max = 3600.,10000

# Full range wave
w_scale = get_wave(eboss_min / (1 + z_max), eboss_max / (1 + z_min))
nbins = len(w_scale)

# Making the truncated grid a nice even number of pixels
nbins_final = 11400
trunc = (nbins - nbins_final) // 2
w_rest = w_scale[trunc:-trunc]
if len(w_rest) == 11401: w_rest = w_rest[:-1] # Odd number fix
l_min = np.log10(w_rest[0])
l_max = np.log10(w_rest[-1])

with fitsio.FITS("qsos.fits") as h:
    fl = h["FLUX"].read()
    w_obs = h["WAVELENGTH"].read()
    redshifts = h["METADATA"].read("Z")

print("Deshifting spectra...")
fl_shift, w_shift = deshift_and_stack(fl, redshifts, w_obs)

print("Adding noise...")
fl_noisy, iv_noisy = add_noise(fl_shift, args.seed, add_noise=args.add_noise)

if args.renorm:
    print("Generating scaling spectra...")  
    spectra_for_norm = generate_scaling_qso(w_scale)
else:
    spectra_for_norm = None
print("Rebinning spectra...")  
fl_rebinned, iv_rebinned, masks = rebin_all(fl_noisy, iv_noisy, w_shift, w_rest, w_scale, spectra_for_norm, renorm=args.renorm, add_noise=args.add_noise)
            
X = np.vstack(fl_rebinned).T
V = np.vstack(iv_rebinned).T

print("post tests")
print(np.where(~X.any(axis=0))[0])
print(np.any(X[:, 4119]))

# Just in case we have a divide by zero error
V = np.nan_to_num(V, nan=0, posinf=0)
X = np.nan_to_num(X, nan=0, posinf=0)

print(np.where(~X.any(axis=0))[0])
print(np.any(X[:, 4119]))

print("Negative fraction:", np.sum(X < 0) / (X[V != 0].size))
print("Missing fraction:", 1 - (np.sum(V != 0) / V.size))

print("Saving spectra...")
if os.path.isfile(args.out): os.remove(args.out)
with fitsio.FITS(args.out, "rw") as h:
    h.write(X, extname="FLUX")
    h.write(V, extname="IVAR")
    h.write(w_rest, extname="WAVELENGTH")
    h.write(redshifts, extname="Z")