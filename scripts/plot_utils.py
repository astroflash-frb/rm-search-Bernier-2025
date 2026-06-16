"""
Functions for plotting Stokes parameters, FDFs, and timing curves.
"""

from pulse_utils import downsample_time_mean
import numpy as np
import astropy.units as u
import colorsys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks


# Default plot parameters
# ------------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 300,
    "figure.figsize": (8, 6),
    "font.size": 12,
    "font.family": "serif",
    "figure.titlesize": 16,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.grid": False,
    "savefig.bbox": "tight",
})


# --- 4-panels plot function ---
def plot_stokes(data_array, freq, time, t_unit='s', suptitle='', 
                rfi_masked=None, height_ratios=[1, 3], cbar_lim_max=None, 
                save_name=None, save_loc=None):
    """
    Plot Stokes parameters I, Q, U, V in a 2x2 grid.

    Parameters
    ----------
    data_array : Array of the 4 Stokes parameters in the order I, Q, U, V. Shape (Nstokes, Ntimes, Nfreqs).
    freq : Frequency array (MHz).
    time : Time array (in t_unit).
    t_unit : Defines the unit for the time array (e.g., 'ms', 's').
    suptitle : Title for the entire figure.
    rfi_masked : List of RFI channels to highlight in red.
    height_ratios : Height ratio of the top/bottom panels for each Stokes subplot.
    cbar_lim_max : Max absolute value for the colorbar limits. If None, limits are set to the 5th and 95th percentiles of Stokes I.
    """
    
    # Figure info
    fig = plt.figure(figsize=(10, 11), dpi=200)
    outer = gridspec.GridSpec(2, 2, hspace=0.15, wspace=0.35)  # 2x2 grid for Stokes parameters
    extent = (time[0], time[-1], freq[0], freq[-1])
    plt.suptitle(suptitle)

    # Go through all Stokes parameters
    for i, param in enumerate(['I', 'Q', 'U', 'V']):
        data_plot = data_array[i]
        row = i // 2
        col = i % 2

        # Define inner grid for timeseries and Stokes parameter plots
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1,  # 2 rows: timeseries + stokes
            subplot_spec=outer[row, col],
            height_ratios=height_ratios,
            hspace=0.0  # no space between timeseries and stokes
        )

        # Add subplots
        timeseries_ax = fig.add_subplot(inner[0])
        stokes_ax = fig.add_subplot(inner[1], sharex=timeseries_ax)

        # --- Add timeseries plot ---
        timeseries = np.nanmean(data_plot, axis=1)  # Average over frequency
        timeseries_ax.scatter(time, timeseries, color='black', s=1)
        timeseries_ax.set_title(f'Stokes {param}')
        timeseries_ax.tick_params(length=0, labelbottom=False, labelleft=False)

        # --- Add Stokes parameter plot ---
        # Colorbar limits
        if cbar_lim_max is None:
            vmin = np.nanpercentile(data_array[0], 5)
            vmax = np.nanpercentile(data_array[0], 95)
        else: 
            vmax = cbar_lim_max
            vmin = -vmax
        linear = plt.cm.colors.Normalize(vmin=vmin, vmax=vmax)
        # Plot data
        im = stokes_ax.imshow(data_plot.T, 
                       origin='lower', 
                       aspect='auto', 
                       extent=extent, 
                       cmap='viridis',  # RdBu
                       norm=linear,
                       )
        
        # --- Colorbar ---
        anchor_height = 0.39 if row==0 else 0.
        cbar = fig.colorbar(im, ax=(timeseries_ax, stokes_ax), pad=0.05, 
                            shrink=height_ratios[1]/np.sum(height_ratios) * 1.06, 
                            anchor=(1.65, anchor_height), extend='both'
                            )
        # cbar.set_label(cbar_label)
        
        # --- Highlight RFI channels ---
        # if rfi_masked is not None:
        #     add_rfi_rectangles(stokes_ax, rfi_masked, freq=freq, time=time)

        # --- Axes labels ---
        if i >= 2:
            stokes_ax.set_xlabel(f'Time [{t_unit}]')
        if i % 2 == 0:
            stokes_ax.set_ylabel('Frequency [MHz]')

    plt.subplots_adjust(top=0.93) 

    # --- Save Figure --
    if save_name is not None and save_loc is not None:
        plt.savefig(save_loc + save_name + ".pdf", dpi=200)
    plt.close()
    

def plot_rmsf(phi_array, RMSF, RMSF_full, save_loc=None, save_name='RMSF'):
    plt.figure(figsize=(8, 5))
    plt.title('RMSF')
    
    plt.plot(phi_array, np.abs(RMSF), lw=1.2, label='w/ Masked Channels')
    plt.plot(phi_array, np.abs(RMSF_full), lw=1, alpha=0.5, c='k', label='All Channels')
    plt.xlim(-500,500)
    
    plt.xlabel(r'$\phi$ [rad/m$^2$]')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')

    if save_name is not None and save_loc is not None:
        plt.savefig(save_loc + save_name + ".png", dpi=300)
    plt.close()



def plot_2panels(data, time, phi, downsamp_factor=8, cbar_label='',
                 ax1_type='peak', ax2_ylabel=r'$\phi$ [rad/m$^2$]', t_unit='s',
                 xlim=None, ylim=None, suptitle=None, save_name=None, save_loc=None):
    """
    Make a 2-panel phi/time plot of some data.

    Parameters
    ----------
    data : Data to plot (shape: [Ntimes, Nphi]). Absolute value is taken.
    time : Time array (shape: [Ntimes]).
    phi : Phi array (shape: [Nphi]).
    cbar_label : Colorbar label.
    ax1_type : Type of the first panel ('peak' for peak amplitude, 'mean' for mean amplitude).
    t_unit : Unit for the time axis (e.g., 'ms', 's').
    """

    # Define figure with 2 rows, 3 columns
    fig = plt.figure(figsize=(8, 8), dpi=150)
    gs = gridspec.GridSpec(2, 3, 
                           height_ratios=[1, 3],
                           width_ratios=[4, 1.2, 0.2],  # main, slice panel, colorbar
                           hspace=0,
                           wspace=0.05)

    ax1 = fig.add_subplot(gs[0, 0])              # timeseries panel
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # main imshow
    ax3 = fig.add_subplot(gs[1, 1], sharey=ax2)  # slice panel
    cax = fig.add_subplot(gs[1, 2])              # colorbar axis
    if suptitle is not None:
        ax1.set_title(suptitle)

    # AX1: True bursts
    # if plot_true:
    #     ax1.axvline(TIME_TRUE[0], color='grey', linestyle='--', label='True bursts')
    #     ax1.axvline(TIME_TRUE[1], color='grey', linestyle='--')
    #     ax1.legend(loc='upper left')

    # AX1: Get peak (or avg) for each time slice
    peak = np.max(np.abs(data), axis=1) if ax1_type=='peak' else np.mean(np.abs(data), axis=1)
    ax1.plot(time, peak, lw=0.8, c='rebeccapurple')
    ax1_label = 'Peak Amp.' if ax1_type == 'peak' else 'Avg. Amp.'
    ax1.set_ylabel(ax1_label)

    # AX2: Plot data (imshow)
    extent = (time[0], time[-1], phi[0], phi[-1])
    im = ax2.imshow(np.abs(data).T,  # shape (Ntimes, Nphi).T
                    aspect='auto',
                    origin='lower',
                    extent=extent,
                    cmap='viridis'
                    )
    ax2.set_xlabel(f'Time [{t_unit}]')
    ax2.set_ylabel(ax2_ylabel)
    if xlim is not None:
        ax2.set_xlim(xlim)
    if ylim is not None:
        ax2.set_ylim(ylim)

    # AX3: downsampled slices (along time axis)
    downsampled_data, _ = downsample_time_mean(data, time, factor=downsamp_factor)
    for d in downsampled_data:
        ax3.plot(abs(d), phi, color='k', alpha=0.1, lw=0.7)
    # ax3.set_xlabel('avg')
    ax3.set_title(f'Time-avg (×{downsamp_factor})')
    ax3.tick_params(labelleft=False)

	# Colorbar
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label)
    
    # --- Save Figure --
    if save_name is not None and save_loc is not None:
        plt.savefig(save_loc + save_name + ".png", dpi=300)
    plt.close()


def plot_cross_corr_slices(phi_lags, cross_corr_arr, time, true_RMs=None,
                           save_name=None, save_loc=None):
    
    plt.figure(figsize=(9, 6))
    plt.title('Cross-Correlation of FDF with RMSF (downsampled by 8)')

   # Downsample & plot data
    if cross_corr_arr.shape[0] <= 8:
        print("Warning: Cross-correlation has very few phi bins, downsampling may not be meaningful.")
    downsampled_cross_corr, _ = downsample_time_mean(cross_corr_arr, time, factor=8)
    for i,t_slice in enumerate(downsampled_cross_corr):
        plt.plot(phi_lags, abs(t_slice), color='k', alpha=0.2, lw=0.7)
    plt.plot([], [], color='black', alpha=1, lw=1, label='FDFs')  # for slice label

    # Vertical lines at true RM (phi) of burst(s)
    num_colors = len(true_RMs)
    tab = 'tab10' if num_colors<=10 else 'tab20'
    cmap = plt.colormaps[tab]
    rm_colors = [cmap(i / num_colors) for i in range(num_colors)]
    
    for i,rm in enumerate(true_RMs):
        plt.axvline(x=rm, label=rf'True $\phi_{i+1}$ = {rm}', color=rm_colors[i], linestyle='--', alpha=0.6)

    # Axes & Labels
    plt.xlabel(r'$\phi$ [rad/m$^2$]')
    plt.ylabel('Amplitude')
    plt.xlim(-700, 700)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.ylim(5, 11)

    if save_name is not None:
        plt.savefig(save_loc + save_name + '.png', dpi=300)
    plt.close()



def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)


def get_colors(num_colors):
    tab = 'tab10' if num_colors<=10 else 'tab20'
    cmap = plt.colormaps[tab]
    
    colors_list = [cmap(i / num_colors) for i in range(num_colors)]
    colors_rgb = [colors.ColorConverter.to_rgb(c) for c in colors_list]
    
    colors_light = [scale_lightness(c, 1.1) for c in colors_rgb]
    colors_dark = [scale_lightness(c, 0.5) for c in colors_rgb]

    return colors_light, colors_dark



def plot_timings_by_block(param_list, metadata_dict, timings_dict, param_name, param_label,
                          fig_title=None, plot_type='cpu_elapsed', save_name=None, save_loc=None,
                          plot_total=True, plot_wall=True, plot_ylog=True, convert_tstep=False):
    """
    Plot timing curves as a function of a tunable parameter for each code block.
    
    Parameters
    ----------
    param_list : list
        List of values for the varied parameter.
    metadata_dict : 
        Dictionary containing info about the data and run parameters. Used for plot title and legend.
    timings_dict : dict
        Dictionary containing timing data for each varied parameter and each code block.
        Structure:
        - Each element of the dict corresponds to a single parameter value. (e.g., timings_dict[param_value])
        - Each element is a dictionary whose keys are code blocks:
            'total' (optional), 'read_and_dedisperse', 'mask_and_normalize', 'rmsf_computation',
            'rm_search', 'cross_correlation'.
        - Each code block maps to another dictionary containing timing metrics:
            'wall_elapsed' : float
                Wall-clock elapsed time.
            'cpu_elapsed' : float
                CPU process time.
            'efficiency' : float
    save_loc : str
        Absolute path where to save the figure.
    param_name : str
        Name of the varied parameter (e.g., 'search_downsamp_factor', 'true_dm_pc_cm3') used in result dicts.
    param_label : str
        Label for the x-axis corresponding to the varied parameter (e.g., 'Downsampling Factor', 'DM [pc/cm^3]').
    plot_type : str
        Type of plot to make (cpu_elapsed or cpu_efficiency). Defaults to 'cpu_elapsed'.
    convert_tstep : bool
        Whether to convert the varied parameter from a downsampling factor to time units (e.g., 391 -> 1ms).
        Only relevant if the varied parameter is the search_downamp_factor.
    """

    # -- Get info for plotting --
    # Data info
    fmin = metadata_dict[param_list[0]]['fmin_MHz']
    fmax = metadata_dict[param_list[0]]['fmax_MHz']
    delta_t_total_s = metadata_dict[param_list[0]]['delta_t_total_s']

    # Convert downsampling factor to a time step in ms for x-axis
    if convert_tstep and param_name=='search_downsamp_factor':
        param_name = "dt_rm_ms"
        param_label = "Search Time Step [ms]"
    
    # Only plot wall times for time curves, not efficiency
    elif plot_type == 'cpu_efficiency':
        plot_wall = False
    
    # Get code block labels and keys for timing dicts
    block_labels = ['Total', 'Read & Dedisperse', 'Mask RFI & Normalize', 'Compute RMSF', 'RM Synthesis', 'Cross-Correlation']
    block_labels = block_labels[1:] if not plot_total else block_labels  # remove 'Total'
    block_keys = list(timings_dict[param_list[0]].keys())[1:] if not plot_total else timings_dict[param_list[0]].keys()
    
    # Get colors
    num_colors = len(block_keys)
    colors_light, colors_dark = get_colors(num_colors)

    # -- Plot --
    fig, ax = plt.subplots(figsize=(8,5))
    fig.suptitle(fig_title)

    # Get and plot data: wall and CPU times OR efficiency
    for i,code_block in enumerate(block_keys):
        data = [timings_dict[p][code_block][plot_type] for p in param_list]
        ax.plot(param_list, data, c=colors_light[i], label=block_labels[i])
        
        if plot_wall:
            wall_data = [timings_dict[p][code_block]['wall_elapsed'] for p in param_list]
            ax.plot(param_list, wall_data,  c=colors_dark[i], ls=':')

    # Axes 
    ax.set_xlabel(param_label)
    ax.set_xlim(param_list[0], param_list[-1])
    ylabel = 'CPU Efficiency' if plot_type=='cpu_efficiency' else 'Time [s]'
    ax.set_ylabel(ylabel)
    if plot_ylog:
        ax.set_yscale('log')

    # -- Legend handles --
    # Color labels (code blocks)
    block_handles = [
        Patch(facecolor=colors_light[i], edgecolor='black', label=block_labels[i])
        for i in range(len(block_keys))
    ]  
    # Line style labels (time type)
    if plot_wall:
        style_handles = [
            Line2D([0], [0], color='black', lw=2, label='CPU Time'),
            Line2D([0], [0], color='black', lw=2, linestyle=':', label='Wall Time')
        ]  
    else:
        style_handles = []
    # Combine handles
    all_handles = (block_handles + style_handles)
    
    # -- Legend --
    fig.canvas.draw()
    bbox = ax.get_position()
    legend_x = bbox.x1 + 0.02
    legend_y = bbox.y1  # top of graph
    
    fig.legend(
        handles=all_handles,
        title=f"{fmin:.1f} - {fmax:.1f} MHz, " + f"{delta_t_total_s:.1f} s",
        loc='upper left',
        bbox_to_anchor=(legend_x, legend_y),
        frameon=True,
        handlelength=2.2,
        labelspacing=0.9
    )

    # -- Save figure --
    if save_name is not None and save_loc is not None:
        plt.savefig(save_loc + save_name, dpi=300)
    plt.close()

