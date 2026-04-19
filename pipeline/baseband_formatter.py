"""
Read raw baseband VDIF files and convert them into full-Stokes data products.

This script assumes the input data are organized using the directory/file format:
    {source_dir}/{fband}/{obs_night}/*_0{fband}.vdif
"""

import argparse
import glob
import os
import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import correlate, correlation_lags
from astropy.constants import c
import astropy.units as u
from astropy.time import Time
import chime_frb_constants as constants
from baseband import vdif
from baseband_tasks.shaping import Transpose, Reshape
from baseband_tasks.combining import Concatenate
from baseband_tasks.functions import Square, Power
from baseband_tasks.dispersion import Dedisperse
from baseband_tasks.dm import DispersionMeasure


# Constants
# ------------------------------------------------------------------------------
SAMPLE_RATE = constants.FPGA_COUNTS_PER_SECOND * u.Hz  # samples per second (time axis sampling)
FCENS = np.linspace(775, 425, 8) # * u.MHz


# Get arguments
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

# Required
parser.add_argument("source_dir", type=Path, help='Path to source directory.')
parser.add_argument("obs_night", type=str, help='Observation night.')
parser.add_argument("save_dir", type=Path, help='Where to save the results.')
parser.add_argument("freq_bands", type=int, nargs='+', help='Frequency bands to process.')

# Optional
parser.add_argument(
    "-i", 
    "--start_file_ind", 
    type=int, 
    default=0, 
    help='Which file index to start at in the source directory.'
)
parser.add_argument(
    "-f", 
    "--nfiles", 
    type=int, 
    default=100, 
    help='Number of files to process.'
)
parser.add_argument(
    "-t", 
    "--num_time_samples", 
    type=int, 
    default=391, 
    help='Number of time samples to average over when reading data. '
)
parser.add_argument(
    "-d",
    "--dispersion_measure",
    type=float,
    default=0.0,
    help='DM to use for dedispersion (in pc/cm^3). Default is 0 (no dedispersion).'
)

args = parser.parse_args()

SOURCE_DIR = args.source_dir
OBS_NIGHT = args.obs_night
SAVE_DIR = args.save_dir
FREQ_BANDS = args.freq_bands
START_FILE_IND = args.start_file_ind
NFILES = args.nfiles
NSAMPLES_PER_CHUNK = args.num_time_samples
DM = DispersionMeasure(args.dispersion_measure * u.pc / u.cm**3)

print(f'\nProcessing source directory: {SOURCE_DIR}')
print(f'Observation night: {OBS_NIGHT}')
print(f'Saving results to: {SAVE_DIR}')
print(f'Frequency bands: {FREQ_BANDS}')
print(f'Starting from file index: {START_FILE_IND}')
print(f'Number of files to process: {NFILES}')
print(f'Number of time samples to average over: {NSAMPLES_PER_CHUNK}')
print(f'Using dispersion measure: {DM}')


# Functions
# ------------------------------------------------------------------------------
def get_freq_array_decreasing(freq_bands):
    """    
    Get frequency array for the specified frequency bands.

    Parameters
    ----------
    freq_bands : list of int
        List of frequency band indices to get the frequency array for. Must be adjacent.
    
    Returns
    -------
    freq_array : np.ndarray
        Frequency array corresponding to the specified frequency bands, shape (Nfreqs,).
        In decreasing frequency order.
    """
    freq_bands = sorted(freq_bands)  # increasing order (so decreasing fcens)
    Nfreqs = int(1024/8 * len(freq_bands))  # number of freq channels within specified bands
    fmin = FCENS[freq_bands[-1]] - 25  # highest band = lowest fcen
    fmax = FCENS[freq_bands[0]] + 25

    freq_array = np.linspace(fmax, fmin, Nfreqs)  # decreasing frequencies
    
    return freq_array


def get_stokes(data):
    """
    Get Stokes I, Q, U, V from powers and cross terms of two polarizations.

    Parameters
    ----------
    data : np.ndarray
        Data array with shape (Ntimes, Nfreqs, Nterms). The Nterms contains the powers
        and cross terms for two polarizations. 
        Axis is ordered as ['XX', 'YY', 'XY', 'YX'] = Re(X X*), Re(Y Y*), Re(X Y*), Im(X Y*).

    Returns
    -------
    Stokes_array : np.ndarray
        Array of Stokes parameters I, Q, U, V, with shape (Nstokes, Ntimes, Nfreqs).
    """
    print("Building Stokes parameters...")
    
    I = data[:,:,0] + data[:,:,1]  # |X|^2 + |Y|^2 = XX* + YY*
    Q = 2*data[:,:,0] - data[:,:,1]  # |X|^2 - |Y|^2 = XX* - YY*
    U = 2 * data[:,:,2]  # 2 Re(XY*)
    V = 2 * data[:,:,3]  # 2 Im(XY*)

    return np.array([I, Q, U, V])


def downsample_data(fh, data, Nsamples_per_chunk=391):
    """
    Downsample complex data along the time axis. 

    Parameters
    ----------
    fh : baseband.vdif.VdifFile
        VDIF file handle from which data is read.
    data : np.ndarray
        Data array matching what is read from fh, with time on axis 0.
    Nsamples_per_chunk : int
        Number of samples the average over. Last chunk will have equal or less than this number.
    
    Returns
    -------
    data_avg : np.ndarray
        Downsampled data array with shape (Nchunks, Nfreqs, Nterms),
        where Nchunks is the number of ~1ms chunks in the file and Nterms are the powers and cross terms (4).
    """
    print("Downsampling data...")

    # Average data along time axis
    Ntimes = fh.shape[0]
    start_edges = list(range(0, Ntimes, Nsamples_per_chunk))  # Start indices for every ~1ms chunk in this file
    data_avg = np.array([np.nanmean(data[e:e+Nsamples_per_chunk], axis=0) for e in start_edges])
    
    return data_avg


def build_time(fh, Nsamples_per_chunk=391):
    """
    Build time array for the data read from a single fh object. This function assumes a 
    list of VDIF files were already concatenated in time

    Parameters
    ----------
    fh : baseband.vdif.VdifFile
        VDIF file handle from which data is read.
    time : List
        List of times since starting to read all files. Gets extended with current file times.
    elapsed_time : float
        Total elapsed time since the start of the time array. Gets updated with current file time.
    """

    # Full time info
    Ntimes = fh.shape[0]
    Dt_total = (fh.stop_time - fh.start_time).sec  # Dt of file
    dt = Dt_total / Ntimes
    time_full = np.arange(Ntimes) * dt  # [s]

    # Downsampled time bins (using center time)
    start_edges = list(range(0, Ntimes, Nsamples_per_chunk))  # Start indices for every ~1ms chunk in this file
    time_downsampled = np.array([np.nanmean(time_full[e:e+Nsamples_per_chunk]) for e in start_edges])

    return time_downsampled


def reshape_fhs(fhs):
    """
    Returns the reshaped fhs as a tuple.
    """
    print("Reshaping...")

    fh_reshaped = []
    for fh in fhs:
        fh = Reshape(fh, (16,2,8))  # splitting frequency and polarization
        fh = Transpose(fh, (1,3,2))  # swapping axes so sub-bands and frequency channels are together
                                     # (Nchannels, Nsubbands, Npols)
        fh = Reshape(fh, (128,2))  # combining sub-bands and frequency channels
        fh_reshaped.append(fh)

    return tuple(fh_reshaped)


def get_data(fhs, freq, dm=0., ref_freq=None, Nsamples_per_chunk=391):
    """
    Read Stokes data from an fh object.

    Parameters
    ----------
    fhs : List of baseband.vdif.VdifFile
        List of VDIF file handle from which data is read. The data inside must have shape
        (Ntimes, Nchannels * Npols, Nsubbands).
    freq : np.ndarray
        Frequency array corresponding to the data in the combined fhs.
        Must be shape (Nfreqs,) and in decreasing order to match fh data.
    dm : DispersionMeasure quantity
        Dispersion measure of the data in fh. Default is 0 (no dedispersion).
    ref_freq : Quantity
        Reference frequency to use for dedispersion. If None, uses the highest frequency 
        in the data. Must be in MHz to match the frequency array.
    Nsamples_per_chunk : int
        Number of time samples to average over when reading data. Default is 391 (corresponding to ~1ms).
    """

    # Reshape data and combine all frequency bands
    fhs = reshape_fhs(fhs)
    print("Concatenating frequency bands...")
    fh = Concatenate(fhs, axis=1)

    # Dedisperse 
    print("De-dispersing...")
    if ref_freq is None:
        ref_freq = np.max(freq)*u.MHz
    fh = Dedisperse(fh, dm=dm, frequency=freq[:, None] * u.MHz,
                    reference_frequency=ref_freq,
                    sideband=1)

    # Convert pol data to power & cross terms
    fh = Power(fh, polarization=['XX', 'YY', 'XY', 'YX'])  # Shape (Ntimes, Nfreqs, 4)
                                # Re(X X*), Re(Y Y*), Re(X Y*), Im(X Y*)
    
    # Read data 
    data = fh.read()  # Shape (Ntimes, Nfreqs, Nterms)
    data_avg = downsample_data(fh, data, Nsamples_per_chunk)  # Downsample to ~1ms (avg every 391 samples)
    stokes_array = get_stokes(data_avg)  # Get all Stokes parameters
    stokes_array = np.flip(stokes_array, axis=2)  # flip frequency axis to increasing order
    print("Data loaded successfully!")
    
    return fh, stokes_array


def read_files(source_dir, obs_night, freq_bands, dm=0., ref_freq=None, 
               start_file_ind=0, Nfiles=100, Nsamples_per_chunk=391):
    """
    Read data from VDIF files in the specified frequency bands and return the Stokes parameters.

    Parameters
    ----------
    source_dir : str
        Directory containing the VDIF files organized by frequency bands.
    obs_night : str
        Directory specifying which observation night to use for this source.
    freq_bands : list of int
        List of frequency band indices to read data from. Indices must give a continuous range of bands.
    start_file_ind : int, optional
        Index of the first file to read in each band. Default is 0.
    Nfiles : int, optional
        Number of files to read from each band. Default is 100.
    Nsamples_per_chunk : int, optional
        Number of time samples to average over when reading data. Default is 391 (corresponding to ~1ms).
    
    Returns
    -------
    stokes_array : np.ndarray
        De-dispersed & downsampled Stokes parameters data for all bands, shape (Nstokes, Ntimes, Nfreqs).
    time : np.ndarray
        Time array corresponding to the data, shape (Ntimes,).
    freq : np.ndarray
        Frequency array corresponding to the data, shape (Nfreqs,).
    """

    # Arrays to return
    freq_bands = sorted(freq_bands)  # increasing band order (so decreasing fcens)
    freq_decrease = get_freq_array_decreasing(freq_bands)  # matches ordering of files [MHz]
    full_data = []  # to store full stokes for all bands
    ref_freq = np.median(freq_decrease) * u.MHz
    print(f'Dedispersion done at a ref freq of {ref_freq}')

    # Get files for all frequency bands (going through bands from largest to lowest fcen)
    fhs = []
    for fband in freq_bands:
        print(f"Getting files for band {fband}...")
        file_list = sorted(glob.glob(f'{source_dir}/{fband}/{obs_night}/*_0{fband}.vdif'))
        file_list = file_list[start_file_ind:start_file_ind+Nfiles] 
        fh = vdif.open(file_list, 'rs', sample_rate=SAMPLE_RATE)  # Open files for this band into a single fh object
        fhs.append(fh)

    # Get data array
    fh, stokes_array = get_data(fhs, freq=freq_decrease, dm=dm, ref_freq=ref_freq,
                                Nsamples_per_chunk=Nsamples_per_chunk)  
                        # shape (Nstokes, Ntimes, Nfreqs) with freqs in increasing order
    time = build_time(fh, Nsamples_per_chunk=Nsamples_per_chunk)  # [seconds]
    freq = np.flip(freq_decrease, axis=0)  # increasing freq array [MHz]

    return stokes_array, time, freq




if __name__ == "__main__":
    # Processing data files
    # ------------------------------------------------------------------------------
    # Get Stokes data ((Nstokes, Ntimes, Nfreqs)
    full_stokes, time, freq = read_files(SOURCE_DIR, OBS_NIGHT,
                                         freq_bands=FREQ_BANDS, 
                                         start_file_ind=START_FILE_IND,
                                         Nfiles=NFILES, dm=DM,
                                         Nsamples_per_chunk=NSAMPLES_PER_CHUNK)

    # Save data
    # ------------------------------------------------------------------------------
    # Create filename and full save path
    source_name = basename = os.path.basename(SOURCE_DIR)
    night_str = OBS_NIGHT.split('_')[0]
    freqs_str = "".join(map(str,FREQ_BANDS))
    save_filename = f'{night_str}_bands{freqs_str}_timeavg{NSAMPLES_PER_CHUNK}_{NFILES}files_start{START_FILE_IND}.npz'
    save_path = os.path.join(SAVE_DIR, source_name, save_filename)

    # Print info about the data being saved
    print(f'Saving data with shape {full_stokes.shape} (Nstokes, Ntimes, Nfreqs)')
    print(f'Time array shape: {time.shape}, Time range: {time[0]} to {time[-1]} seconds')
    print(f'Time downsampled to ~{(NSAMPLES_PER_CHUNK / SAMPLE_RATE).to(u.ms):.2f} ({NSAMPLES_PER_CHUNK} samples)')
    print(f'Frequency array shape: {freq.shape}, Frequency range: {freq[0]} to {freq[-1]} MHz')
    print(f'Saving to: {save_path}')

    # Save data as .npz file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, 
                        full_stokes=full_stokes, 
                        time=time,
                        freq=freq)