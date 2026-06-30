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
                cbar_lim_max=None, save_name=None, save_loc=None):
    """
    Plot Stokes parameters I, Q, U, V in a 2x2 grid.

    Parameters
    ----------
    data_array : np.ndarray
        Array of the 4 Stokes parameters in the order I, Q, U, V. Shape (Nstokes, Ntimes, Nfreqs).
    freq : np.ndarray
        Frequency array (MHz).
    time : np.ndarray
        Time array (in t_unit).
    t_unit : string
        Defines the unit for the time array (e.g., 'ms', 's').
    suptitle : string
        Title for the entire figure.
    cbar_lim_max : 
        Max absolute value for the colorbar limits. If None, limits are set to the 5th and 95th percentiles of Stokes I.
    save_name : str, optional
        Name for the saved figure (without extension). If None, the figure is not saved.
    save_loc : str, optional
        Absolute path to the directory where the figure should be saved. If None, the figure is not saved.
    """
    
    # Figure info
    fig = plt.figure(figsize=(10, 11), dpi=150)
    height_ratios=[1, 3]
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

        # --- Axes labels ---
        if i >= 2:
            stokes_ax.set_xlabel(f'Time [{t_unit}]')
        if i % 2 == 0:
            stokes_ax.set_ylabel('Frequency [MHz]')

    plt.subplots_adjust(top=0.93) 

    # --- Save Figure --
    if save_name is not None and save_loc is not None:
        plt.savefig(save_loc + save_name + ".pdf", dpi=150)
    plt.close()
    

def plot_rmsf(phi_array, RMSF, RMSF_full, save_loc=None, save_name='RMSF'):
    """
    Plot the RMSF (Rotation Measure Spread Function) with and without masked channels.

    Parameters
    ----------
    phi_array : np.ndarray
        Array of phi values (rad/m^2).
    RMSF : np.ndarray
        RMSF computed with masked channels.
    RMSF_full : np.ndarray
        RMSF computed with all channels (no masking).
    save_loc : str, optional
        Absolute path to the directory where the figure should be saved. If None, the figure is not saved.
    save_name : str, optional
        Name for the saved figure (without extension). Defaults to 'RMSF'.
    """

    plt.figure(figsize=(8, 5))
    plt.title('RMSF')
    
    # Plot RMSF with and without masked channels
    plt.plot(phi_array, np.abs(RMSF), lw=1.2, label='w/ Masked Channels')
    plt.plot(phi_array, np.abs(RMSF_full), lw=1, alpha=0.5, c='k', label='All Channels')
    plt.xlim(-500,500)
    
    # Axes & Labels
    plt.xlabel(r'$\phi$ [rad/m$^2$]')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')

    # Save Figure
    if save_name is not None and save_loc is not None:
        plt.savefig(save_loc + save_name + ".png", dpi=150)
    plt.close()



def plot_2panels(data, time, phi, downsamp_factor=8, cbar_label='',
                 ax2_ylabel=r'$\phi$ [rad/m$^2$]', t_unit='s',
                 xlim=None, ylim=None, suptitle=None, save_name=None, save_loc=None):
    """
    Make a 2-panel phi/time waterfall plot of some data. Top panel shows the peak amplitude in the timeseries.

    Parameters
    ----------
    data : np.ndarray
        Data to plot (shape: [Ntimes, Nphi]). Absolute value is taken.
    time : np.ndarray
        Time array (shape: [Ntimes]).
    phi : np.ndarray
        Phi array (shape: [Nphi]).
    cbar_label : str
        Colorbar label.
    ax2_ylabel : str
        Y-axis label for the main imshow panel (bottom).
    t_unit : 
        Unit for the time axis (e.g., 'ms', 's').
    xlim : tuple, optional
        X-axis limits for the main imshow panel (bottom). If None, limits are set to the min and max of the time array.
    ylim : tuple, optional
        Y-axis limits for the main imshow panel (bottom). If None, limits are set to the min and max of the phi array.
    suptitle : str, optional
        Title for the entire figure.
    save_name : str, optional
        Name for the saved figure (without extension). If None, the figure is not saved.
    save_loc : str, optional
        Absolute path to the directory where the figure should be saved. If None, the figure is not saved.
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
    peak = np.max(np.abs(data), axis=1)
    ax1.plot(time, peak, lw=0.8, c='rebeccapurple')
    ax1.set_ylabel('Peak Amp.' )

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
        plt.savefig(save_loc + save_name + ".png", dpi=150)
    plt.close()


def plot_cross_corr_slices(phi_lags, cross_corr_arr, time, true_RMs=None,
                           save_name=None, save_loc=None):
    """
    Plot the cross-correlation of the FDF with the RMSF, downsampled by a factor of 8, with vertical lines at the true RM values.
    
    Parameters
    ----------
    phi_lags : np.ndarray
        Array of phi lag values (rad/m^2).
    cross_corr_arr : np.ndarray
        Array of cross-correlation values (shape: [Ntimes, Nphi_lags]).
    time : np.ndarray
        Array of time values corresponding to the time axis in cross_corr_arr.
    true_RMs : list or np.ndarray, optional
        List or array of true RM values (rad/m^2) to plot as vertical lines. If None, no vertical lines are plotted.
    save_name : str, optional
        Name for the saved figure (without extension). If None, the figure is not saved.
    save_loc : str, optional
        Absolute path to the directory where the figure should be saved. If None, the figure is not saved.
    """
    
    plt.figure(figsize=(9, 6))
    plt.title('Cross-Correlation of FDF with RMSF (downsampled by 8)')

   # Downsample data
    if cross_corr_arr.shape[0] <= 8:
        print("Warning: Cross-correlation has very few time bins, downsampling may not be meaningful.")
    downsampled_cross_corr, _ = downsample_time_mean(cross_corr_arr, time, factor=8)
    
    # Plot each downsampled slice
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
        plt.savefig(save_loc + save_name + '.png', dpi=150)
    plt.close()



def scale_lightness(rgb, scale_l):
    """
    Helper function to scale the lightness of an RGB color by a given factor.
    """

    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)


def get_colors(num_colors):
    """
    Helper function to get a list of colors for plotting, with light and dark variants for each color.
    """

    tab = 'tab10' if num_colors<=10 else 'tab20'
    cmap = plt.colormaps[tab]
    
    colors_list = [cmap(i / num_colors) for i in range(num_colors)]
    colors_rgb = [colors.ColorConverter.to_rgb(c) for c in colors_list]
    
    colors_light = [scale_lightness(c, 1.1) for c in colors_rgb]
    colors_dark = [scale_lightness(c, 0.5) for c in colors_rgb]

    return colors_light, colors_dark


def plot_timings_by_block(param_list, metadata_dict, timings_dict, param_label,
                          fig_title=None, plot_type='cpu_elapsed', save_name=None, save_loc=None,
                          plot_total=True, plot_wall=True, plot_ylog=True):
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
    param_label : str
        Label for the x-axis corresponding to the varied parameter (e.g., 'Downsampling Factor', 'DM [pc/cm^3]').
    plot_type : str
        Type of plot to make (cpu_elapsed or cpu_efficiency). Defaults to 'cpu_elapsed'.
    """

    # -- Get info for plotting --
    # Data info
    fmin = metadata_dict[param_list[0]]['fmin_MHz']
    fmax = metadata_dict[param_list[0]]['fmax_MHz']
    delta_t_total_s = metadata_dict[param_list[0]]['delta_t_total_s']

    # Only plot wall times for time curves, not efficiency
    if plot_type == 'cpu_efficiency':
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
        plt.savefig(save_loc + save_name + '.png', dpi=150)
    plt.close()



def plot_bursts_in_fdf(p, stokes_I, time_arr, phi_arr, visible_dict, downsamp_list,
                       results_params_all_downsamp, param_label, rm_true,
                       save_name=None, save_loc=None):
    """
    Produce a 2-panel plot showing the S/N of Stokes I over time (bottom panel) and 
    the detected bursts in the FDF (top panel) for a single parameter value of the RM search. 
    The top panel of the plot contains results for all results of the pulse detection (eg. varied pulse downsampling factor).

    Parameters
    ----------
    p : float
        The value of the varied parameter in the RM search for which to plot the bursts.
    stokes_I : np.ndarray
        2D array of Stokes I values (shape: [Ntimes, Nfreqs]).
    time_arr : np.ndarray
        Time values corresponding to the Stokes I data. 
        This does not have to match the time axis of the FDF.
    phi_arr : np.ndarray
        Array of phi values (rad/m^2) corresponding to the FDF.
    visible_dict : dict
        Dictionary containing information about which bursts are visible for each parameter value.
    downsamp_list : list
        List of downsampling factors used in 02_pulse_detection.py that we want to plot.
    results_params_all_downsamp : list of dicts
        List of dictionaries containing the results for each downsampling factor.
        Each dictionary corresponds to the "params" results for the output of 
        02_pulse_detection.py for a given downsampling factor. See the structure of the output of 02_pulse_detection.py for details.
    param_label : str
        Label for the varied parameter in the RM search.
    rm_true : list or np.ndarray
        List or array of true RM values (rad/m^2) to plot as horizontal lines in the top panel.
    save_name : str, optional
        Name for the saved figure (without extension). If None, the figure is not saved.
    save_loc : str, optional
        Absolute path to the directory where the figure should be saved. If None, the figure is not saved.
    """

    # ----- Define figure -----
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]}
    )
    fig.suptitle(f"{param_label} = {p:.0f}", y=0.95)

    # ----- Compute & Plot Stokes Timeseries -----
    timeseries = np.nanmean(stokes_I, axis=1)  # Average over frequency
    median = np.nanmedian(timeseries)
    std = np.nanstd(timeseries)
    snr_timeseries = (timeseries - median) / std

    ax_bot.scatter(time_arr, snr_timeseries, 
                   color='black', s=1, zorder=1,
                   label='Stokes I')

    # ----- Get & Plot Visible Pulses -----
    visible_scatter_idx = visible_dict[p]['visible_bursts']
    Nvis = visible_dict[p]['num_pulses_visible']

    ax_bot.scatter(time_arr[visible_scatter_idx], 
                   snr_timeseries[visible_scatter_idx],
                   color='red', s=5, zorder=3,
                   label='Visible Pulses'
                  )
                   
    for v in visible_scatter_idx:
        ax_bot.axvspan(xmin=time_arr[v]-0.1, 
                       xmax=time_arr[v]+0.1,
                       color='red', alpha=0.1, zorder=2,
                       linewidth=0
                      )
    
    # ----- Colors for Downsampling Factors -----
    num_colors = len(downsamp_list)
    cmap = plt.get_cmap('tab10' if num_colors <= 10 else 'tab20')
    colors = [cmap(i) for i in range(num_colors)]
    markers = [4,5,6,7,'o','+','d', 's']
    
    for i,downsamp in enumerate(downsamp_list):
        # ----- Get All Bursts Info -----
        allbursts = results_params_all_downsamp[i][p]['detected_bursts']
        notvis_intervals = set(results_params_all_downsamp[i][p]['notvisible_intervals'])
        Ndetec = len(allbursts)
        Nnotvis = len(notvis_intervals)

        # --- Plot Bursts individually -----
        for burst in allbursts:
            burst_times = []
            phi_mean = []

            # get info for each detection of a burst
            for b in burst:
                t = b["time"]
                peaks = b["peaks"]

                burst_times.append(t)
                phi_mean.append(np.mean(phi_arr[peaks])) 

            # check if classified as visible or not in Stokes I
            t_interval = burst_times[0], burst_times[-1]
            notvis_flag = t_interval in notvis_intervals
            m = "x" if notvis_flag else markers[i]
            lw = 1.5 if notvis_flag else 0
            alpha = 1 if notvis_flag else 0.5
            s = 12 if notvis_flag else 25
            
            # Add all burst data to top panel
            ax_top.scatter(np.mean(burst_times), np.mean(phi_mean), 
                           s=s, marker=m, color=colors[i], 
                           lw=lw, alpha=alpha, 
                           label=f"downsamp factor = {downsamp:.0f}")
         
    # ----- Plot True RM -----
    for rm in rm_true:
        ax_top.axhline(y=rm, ls=':', color='grey', lw=1)
        
    # ----- Legends -----
    ax_bot.legend(title=f"Nvis = {Nvis}", loc="center left", bbox_to_anchor=(1, 0.5))
    ax_top.legend(
        title=f"Ndetec = {Ndetec}",
        loc="center left", bbox_to_anchor=(1, 0.5),
        title_fontsize = 9, fontsize=9,
        handles=[
        *[
            Line2D([0],[0], marker=markers[i], color='none',
                   markerfacecolor=colors[i], markeredgecolor=colors[i],
                   linestyle='None', label=f"downsamp factor = {downsamp_list[i]:.0f}")
            for i in range(len(downsamp_list))
        ],
        Line2D([0],[0], color='grey', ls=':', label='True RM'),
        Line2D([0],[0], marker='x', color='black',
               linestyle='None', label='Not visible')
    ])
    
    # ----- Axes -----
    ax_top.set_ylabel(r'$\phi$ [rad/m$^2$]')
    ax_bot.set_xlabel("Time [s]")
    ax_bot.set_ylabel("S/N")
    ax_bot.set_ylim(np.min(snr_timeseries)-0.5, np.max(snr_timeseries)+0.5)
    for ax in [ax_top, ax_bot]:
        ax.set_xlim(np.min(time_arr)-0.1, np.max(time_arr)+0.1)
    
    # ----- Save figure -----
    fig.tight_layout()
    if save_name is not None and save_loc is not None:
        plt.savefig(save_loc + save_name + ".png", dpi=150)
    plt.close()