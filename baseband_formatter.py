"""
Read raw baseband VDIF files and convert them into full-Stokes data products.

This script assumes the input data are organized using the directory/file format:
    {source_dir}/{fband}/{obs_night}/*_0{fband}.vdif
"""

import argparse
from pathlib import Path
import glob
import os
import numpy as np
import astropy.units as u
import chime_frb_constants as constants
from baseband import vdif
from baseband_tasks.shaping import Transpose, Reshape
from baseband_tasks.functions import Power


# Constants
# ------------------------------------------------------------------------------
SAMPLE_RATE = constants.FPGA_COUNTS_PER_SECOND * u.Hz
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

args = parser.parse_args()

SOURCE_DIR = args.source_dir
OBS_NIGHT = args.obs_night
SAVE_DIR = args.save_dir
FREQ_BANDS = args.freq_bands
START_FILE_IND = args.start_file_ind
NFILES = args.nfiles
NSAMPLES_PER_CHUNK = args.num_time_samples

print(f'\nProcessing source directory: {SOURCE_DIR}')
print(f'Observation night: {OBS_NIGHT}')
print(f'Saving results to: {SAVE_DIR}')
print(f'Frequency bands: {FREQ_BANDS}')
print(f'Starting from file index: {START_FILE_IND}')
print(f'Number of files to process: {NFILES}')
print(f'Number of time samples to average over: {NSAMPLES_PER_CHUNK}')


# Functions
# ------------------------------------------------------------------------------
def get_freq_array(freq_bands):
    """    
    Get frequency array for the specified frequency bands.

    Parameters
    ----------
    freq_bands : list of int
        List of frequency band indices to get the frequency array for. 
        Must be adjacent and sorted in increasing band order (decreasing fcen order).
    
    Returns
    -------
    freq_array : np.ndarray
        Frequency array (increasing) corresponding to the specified frequency bands, shape (Nfreqs,).
    """

    Nfreqs = int(1024/8 * len(freq_bands))  # number of freq channels within specified bands
    fstart = FCENS[freq_bands[-1]] - 25  # highest band = lowest fcen
    fstop = FCENS[freq_bands[0]] + 25

    freq_array = np.linspace(fstart, fstop, Nfreqs)
    
    return freq_array


def get_stokes(data):
    """
    Get Stokes I, Q, U, V from powers and cross terms of two polarizations.

    Parameters
    ----------
    data : np.ndarray
        Data array with shape (Ntimes, Nfreqs, Nterms). The Nterms contains the powers
        and cross terms for two polarizations. Axis is ordered as ['XX', 'YY', 'XY', 'YX'].

    Returns
    -------
    I, Q, U, V : np.ndarrays
        Stokes parameters I, Q, U, V, each with shape (Ntimes, Nfreqs).
    """
    
    I = data[:,:,0] + data[:,:,1]  # |X|^2 + |Y|^2
    Q = data[:,:,0] - data[:,:,1]  # |X|^2 - |Y|^2
    U = 2 * data[:,:,2]  # 2 Re(XY*)
    V = 2 * data[:,:,3]  # 2 Im(XY*)

    return I, Q, U, V


def downsample_data(fh, data, Nsamples_per_chunk=391):
    """
    Downsample data along the time axis. 

    Parameters
    ----------
    fh : baseband.vdif.VdifFile
        VDIF file handle from which data is read.
    data : np.ndarray
        Data array matching what is read from fh.
    Nsamples_per_chunk : int
        Number of samples the average over. Last chunk will have equal or less than this number.
    
    Returns
    -------
    data_avg : np.ndarray
        Downsampled data array with shape (Nchunks, Nfreqs, Nterms),
        where Nchunks is the number of ~1ms chunks in the file and Nterms are the powers and cross terms (4).
    """

    # Average data along time axis
    Ntimes = fh.shape[0]
    start_edges = list(range(0, Ntimes, Nsamples_per_chunk))  # Start indices for every ~1ms chunk in this file
    data_avg = np.array([np.nanmean(data[e:e+Nsamples_per_chunk], axis=0) for e in start_edges])
    
    return data_avg


def build_time(fh, time, elapsed_time, Nsamples_per_chunk=391):
    """
    Build time array for the data read from a single VDIF file. This function updates the global time array
    and the elapsed time since the start of the global time array.

    Parameters
    ----------
    fh : baseband.vdif.VdifFile
        VDIF file handle from which data is read.
    time : List
        List of times since starting to read all files. Gets extended with current file times.
    elapsed_time : float
        Total elapsed time since the start of the time array. Gets updated with current file time.
    """

    Ntimes = fh.shape[0]
    start_edges = list(range(0, Ntimes, Nsamples_per_chunk))  # Start indices for every ~1ms chunk in this file
    
    # File time info
    dt_total = (fh.stop_time - fh.start_time).sec  # dt of file
    t_file = np.linspace(0, dt_total, Ntimes)  # relative times within file, [s]

    # Time of 1ms chunks relative to start of time array
    t_data = [t_file[e] + elapsed_time for e in start_edges]
    time.extend(t_data)

    # Update time
    elapsed_time = elapsed_time + dt_total
    return elapsed_time


def get_data(fh, data_list, transpose_shape=(1,3,2), Nsamples_per_chunk=391):
    """
    Read data from a single VDIF file and append to data_list.

    Parameters
    ----------
    fh : baseband.vdif.VdifFile
        VDIF file handle from which data is read. The data inside must have shape
        (Ntimes, Nchannels * Npols, Nsubbands).
    data_list : list of np.ndarray
        List containing data arrays for each opened file. List has length Nfiles, 
        and each array has shape [Nstokes, Ntimes, Nfreqs].
    transpose_shape : tuple of int
        Order of axis indices to use for transposing the loaded data when reshaping 
        to merge frequency axes. The time axis is exluded. 
        Default transposes to (Nchannels, Nsubbands, Npols).
    """

    # Reshape the data
    fh = Reshape(fh, (16,2,8))  # splitting frequency and polarization
    fh = Transpose(fh, transpose_shape)  # swapping axes so sub-bands and frequency channels are together
    fh = Reshape(fh, (128,2))  # combining sub-bands and frequency channels

    # Convert pol data to power & cross terms
    fh = Power(fh, polarization=['XX', 'YY', 'XY', 'YX'])  # Shape (Ntimes, Nfreqs, 4)
    
    # Read data
    data = fh.read()  # Shape (Ntimes, Nfreqs, Nterms)

    # Downsample to ~1ms (avg every 391 samples)
    data_avg = downsample_data(fh, data, Nsamples_per_chunk)

    # Get all Stokes parameters
    I, Q, U, V = get_stokes(data_avg)
    # Add to list of data for this band
    stokes_stack = np.array([I, Q, U, V])  # Shape: (4, Ntimes, Nfreqs)
    data_list.append(stokes_stack)


def read_files(source_dir, obs_night, freq_bands, start_file_ind=0, Nfiles=100, Nsamples_per_chunk=391):
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
    
    Returns
    -------
    full_data : np.ndarray
        Downsampled Stokes parameters data for all bands, shape (Nstokes, Ntimes, Nfreqs).
    time : np.ndarray
        Time array corresponding to the data, shape (Ntimes,).
    freq : np.ndarray
        Frequency array corresponding to the data, shape (Nfreqs,).
    """

    # Arrays to return
    time = []  # [seconds]
    freq_bands = sorted(freq_bands)  # increasing order
    freq = get_freq_array(freq_bands)  # [MHz]
    full_data = []

    elapsed_time = 0.0  # [seconds]

    for band_index,fband in enumerate(freq_bands):
        print(f'Processing band {fband}')
        
        file_list = sorted(glob.glob(f'{source_dir}/{fband}/{obs_night}/*_0{fband}.vdif'))
        band_data = []  # Stores I,Q,U,V for this band

        for file in file_list[start_file_ind:start_file_ind+Nfiles]:
            # Open file
            fh = vdif.open(file, 'rs', sample_rate=SAMPLE_RATE)  

            # Add file data to band_data (Nfiles * array[Nstokes, Ntimes_file, Nfreqs])
            get_data(fh, band_data, Nsamples_per_chunk=Nsamples_per_chunk)

            # Build time array (only for first band)
            if band_index==0:
                elapsed_time = build_time(fh, time, elapsed_time, 
                                          Nsamples_per_chunk=Nsamples_per_chunk)
            
        # Reshape to (Nstokes, Ntimes, Nfreqs), all increasing
        band_data = np.concatenate(band_data, axis=1)  # Concatenate along time axis
        band_data = np.flip(band_data, axis=2)  # flip frequency axis to increasing order
        # Add to full_data
        full_data.append(band_data)

    # Flip band order & concatenate along frequency axis
    full_data = np.concatenate(full_data[::-1], axis=2)
    # Time array
    time = np.array(time)  # [s]

    return full_data, time, freq


if __name__ == "__main__":
    # Processing data files
    # ------------------------------------------------------------------------------
    # Get Stokes data ((Nstokes, Ntimes, Nfreqs)
    full_stokes, time, freq = read_files(SOURCE_DIR, OBS_NIGHT,
                                        freq_bands=FREQ_BANDS, 
                                        start_file_ind=START_FILE_IND,
                                        Nfiles=NFILES, 
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
    print(f'Time samples averaged over every {NSAMPLES_PER_CHUNK} samples (~{(NSAMPLES_PER_CHUNK / SAMPLE_RATE).to(u.ms):.2f})')
    print(f'Frequency array shape: {freq.shape}, Frequency range: {freq[0]} to {freq[-1]} MHz')
    print(f'Saving to: {save_path}')

    # Save data as .npz file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, 
                        full_stokes=full_stokes, 
                        time=time,
                        freq=freq)