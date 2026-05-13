"""
Functions for plotting Stokes parameters, FDFs, and timing curves.
"""

import numpy as np
import astropy.units as u
import colorsys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmcrameri.cm as cmc
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
        plt.savefig(save_loc + save_name + ".pdf", dpi=300)
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
        plt.savefig(save_loc + save_name, dpi=300)
    plt.close()


def downsample_time_mean(data, factor):
    """
    Downsample data along the time axis by some factor (number of time bins per group)
    """
    # Trim time axis so it's divisible by 'factor'
    Ntime = data.shape[0]
    Ntime_trimmed = (Ntime // factor) * factor
    trimmed_data = data[:Ntime_trimmed]

    # Group consecutive time samples into bins of size 'factor'
    # New shape: (Ngroups, factor, Nphi)
    grouped_data = trimmed_data.reshape(-1, factor, data.shape[1])

    # Average within each group (collapse the 'factor' axis)
    downsampled_data = grouped_data.mean(axis=1)
    
    return downsampled_data
    

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
    downsampled_data = downsample_time_mean(data, factor=downsamp_factor)
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
        plt.savefig(save_loc + save_name, dpi=300)
    plt.close()


def plot_folded_fdf(phi_arr, fdf_arr, lim_inds, true_RMs=None, 
                    xmax=None, save_name=None, save_loc=None):
    # Fold FDF around phi=0 by summing positive and negative phi sides
    if fdf_arr.shape[0] <= 8:
        print("Warning: FDF has very few phi bins, folding may not be meaningful.")
    fdf_slice = abs(np.nanmean(fdf_arr[lim_inds[0]:lim_inds[1]], axis=0))
    phi_zero_ind = np.argwhere(phi_arr==0)[0,0]  # index of phi=0 in phi_arr
    folded_phi = phi_arr[phi_zero_ind:]
    pos_phi_fdf = fdf_slice[phi_zero_ind:]
    neg_phi_fdf = np.concatenate([[0], np.flip(fdf_slice[:phi_zero_ind])])
    folded_fdf = (pos_phi_fdf + neg_phi_fdf)  # using sum

    fig = plt.figure(figsize=(8,3))
    
    # FDF lines
    pos_line, = plt.plot(folded_phi, pos_phi_fdf, lw=1, ls=":", c="k", alpha=0.7, label=f"$+\phi$ range")  # positive phi
    neg_line, = plt.plot(folded_phi, neg_phi_fdf, lw=1, ls="--", c="k", alpha=0.7, label=f"$-\phi$ range")  # negative phi
    combined_line, = plt.plot(folded_phi, folded_fdf, lw=1, c="rebeccapurple", label=f"Sum")  # sum
    
    for i,rm in enumerate(true_RMs):
        plt.axvline(x=abs(rm), label=rf'True $\phi_{i+1}$ = {rm}', c='dodgerblue', linestyle='--', alpha=0.6)
    plt.axvspan(-10, 10, alpha=0.5, color='lightgrey')

    # Axes & Labels
    plt.legend(handles=[combined_line, pos_line, neg_line], loc="upper right")
    fig.supylabel("FDF Amplitude")
    fig.supxlabel(r'|$\phi$| [rad/m$^2$]')
    if xmax is not None:
        plt.xlim(0,xmax)

    # Save figure
    plt.tight_layout()
    if save_name is not None and save_loc is not None:
        plt.savefig(save_loc + save_name, dpi=200)
    plt.close()



def plot_cross_corr_slices(phi_lags, cross_corr_arr, true_RMs=None,
                           save_name=None, save_loc=None):
    
    plt.figure(figsize=(9, 6))
    plt.title('Cross-Correlation of FDF with RMSF (downsampled by 8)')

   # Downsample & plot data
    if cross_corr_arr.shape[0] <= 8:
        print("Warning: Cross-correlation has very few phi bins, downsampling may not be meaningful.")
    downsampled_cross_corr = downsample_time_mean(cross_corr_arr, factor=8)
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
        plt.savefig(save_loc + save_name, dpi=300)
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


def plot_time_curves_by_block(param_arr, metadata_dict_list, timings_dict_list,
                              x_label, fig_title=None, save_name=None, save_loc=None,
                              plot_total=True, plot_wall=True, plot_ylog=True, convert_tstep=False):
    """
    Plot timing curves as a function of a tunable parameter for each code block.
    
    Parameters
    ----------
    dict_list : list of dicts
        List of info for each run with a different parameter value. Contains a timings dict
    timing_dicts : list of dict
        Collection of timing data, one entry per value of the tunable parameter.
    
        Structure:
        - Each element of the list corresponds to a single parameter value.
        - Each element is a dictionary whose keys are code blocks:
            'total' (optional), 'load_mask_normalize', 'rmsf_computation',
            'rm_search', 'cross_correlation'.
        - Each code block maps to another dictionary containing timing metrics:
            'wall_elapsed' : float
                Wall-clock elapsed time.
            'cpu_elapsed' : float
                CPU process time.
    save_loc : str
        Absolute path where to save the figure.
    """

    # -- Info for plotting --
    data_info = metadata_dict_list[0]  # using first dict (they all use the same metadata except the varied param)
    
    # Convert search time step to time units
    if convert_tstep :
        param_arr = [(d['dt_rm_ms']).to(u.ms).value for d in metadata_dict_list]
        x_label = "Search Time Step [ms]"
    
    # Get code block labels and keys for timing dicts
    block_labels = ['Total', 'Mask RFI & Normalize', 'RMSF', 'RM Search', 'Cross-Correlation']
    block_labels = block_labels[1:] if not plot_total else block_labels  # remove 'Total'
    block_keys = list(timings_dict_list[0].keys())[1:] if not plot_total else timings_dict_list[0].keys()

    # Get colors
    num_colors = len(block_keys)
    colors_light, colors_dark = get_colors(num_colors)

    # -- Plot --
    fig, ax = plt.subplots(figsize=(8,5))
    fig.suptitle(fig_title)

    # Get and plot data: wall and CPU times
    for i,code_block in enumerate(block_keys):
        cpu_data = [t_dict[code_block]['cpu_elapsed'] for t_dict in timings_dict_list]
        ax.plot(param_arr, cpu_data, c=colors_light[i], label=block_labels[i])
        
        if plot_wall:
            wall_data = [t_dict[code_block]['wall_elapsed'] for t_dict in timings_dict_list]
            ax.plot(param_arr, wall_data,  c=colors_dark[i], ls=':')

    # Axes 
    ax.set_xlabel(x_label)
    ax.set_ylabel('Time [s]')
    if plot_ylog:
        ax.set_yscale('log')

    # -- Legend handles --
    # Color labels (code blocks)
    block_handles = [
        Patch(facecolor=colors_light[i], edgecolor='black', label=block_labels[i])
        for i in range(len(block_keys))
    ]  
    # Line style labels (time type)
    style_handles = [
        # Line2D([0], [0], color='black', lw=2, label='CPU Time'),
        # Line2D([0], [0], color='black', lw=2, linestyle=':', label='Wall Time')
    ]  
    # Combine handles
    all_handles = (block_handles + style_handles)
    
    # -- Legend --
    fig.canvas.draw()
    bbox = ax.get_position()
    legend_x = bbox.x1 + 0.02
    legend_y = bbox.y1  # top of graph
    
    fig.legend(
        handles=all_handles,
        title=f"{data_info['fmin']:.1f} - {data_info['fmax']:.1f} MHz, " + 
                f"{data_info['delta_t_total_s']:.1f} s",
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


def plot_efficiency_by_block(param_arr, metadata_dict_list, timings_dict_list, 
                             x_label, fig_title=None, save_name=None, save_loc=None, 
                             plot_total=True, plot_ylog=True, convert_tstep=False):
    """
    Plot CPU efficiency curves as a function of a tunable parameter for each code block.
    
    Parameters
    ----------
    dict_list : list of dicts
        List of info for each run with a different parameter value. Contains a timings dict
    timing_dicts : list of dict
        Collection of timing data, one entry per value of the tunable parameter.
    
        Structure:
        - Each element of the list corresponds to a single parameter value.
        - Each element is a dictionary whose keys are code blocks:
            'total' (optional), 'load_mask_normalize', 'rmsf_computation',
            'rm_search', 'cross_correlation'.
        - Each code block maps to another dictionary containing timing metrics:
            'wall_elapsed' : float
                Wall-clock elapsed time.
            'cpu_elapsed' : float
                CPU process time.
    save_loc : str
        Absolute path where to save the figure.
    """

    # -- Info for plotting --
    data_info = metadata_dict_list[0]  # using first dict (they all use the same data)
    
    # Convert search time step to time
    if convert_tstep :
        param_arr = [(d['dt_rm']).to(u.ms).value for d in metadata_dict_list]
        x_label = "Search Time Step [ms]"
        
    # Get code block labels and keys for timing dicts
    block_labels = ['Total', 'Mask RFI & Normalize', 'RMSF', 'RM Search', 'Cross-Correlation']
    block_labels = block_labels[1:] if not plot_total else block_labels  # remove 'Total'
    block_keys = list(timings_dict_list[0].keys())[1:] if not plot_total else timings_dict_list[0].keys()

    # Get colors
    num_colors = len(block_keys)
    colors_light, _ = get_colors(num_colors)

    # -- Plot --
    fig, ax = plt.subplots(figsize=(8,5))
    fig.suptitle(fig_title)

    # Get and plot data
    for i,code_block in enumerate(block_keys):
        data = [t_dict[code_block]['cpu_efficiency'] for t_dict in timings_dict_list]
        ax.plot(param_arr, data, c=colors_light[i], label=block_labels[i])

    # Axes
    ax.set_xlabel(x_label)
    ax.set_ylabel('CPU Efficiency')
    if plot_ylog:
        ax.set_yscale('log')

    # -- Legend --
    fig.canvas.draw()
    bbox = ax.get_position()
    legend_x = bbox.x1 + 0.02
    legend_y = bbox.y1  # top of graph
    
    fig.legend(
        title=f"{data_info['fmin']:.1f} - {data_info['fmax']:.1f} MHz, " + 
                f"{data_info['delta_t_total_s']:.1f} s",
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



def get_max_snr(data, phi_arr, time, t_true, rm_true):
    """
    Compute SNR of peaks in the FDF (or cross-correlation) where the bursts occur.
    SNR at each time is calculated using (peak_signal - baseline) / sigma_noise, where
    peak_signal is the FDF value at the true rm, baseline is the median of the time slice with
    the full peak masked, and sigma noise is the noise std for the given time bin.
    The function returns the peak SNR in time for each burst.

    Parameters
    ----------
    data : np.ndarray
        data of shape (Ntime, Nphi)
    phi_arr : 
        must match the phi axis of data
    time : 
        must match time axis of data
    t_true : list of floats
        list of times at which bursts occur
    rm_true : list of floats
        list of true RMs of bursts. Ordering must match t_true
    """

    snr_max_per_peak = []
    
    for i,rm in enumerate(rm_true):
        rm_ind = np.argmin(np.abs(phi_arr - rm))  # index of true RM along phi array
        mask_start, mask_stop = np.argmin(abs(phi_arr-(rm-125))), np.argmin(abs(phi_arr-(rm+125)))  # phi edges to mask burst
        t1,t2 = np.argmin(abs(time-(t_true[i]-0.3))), np.argmin(abs(time-(t_true[i]+0.3)))  # time range to compute SNR for this burst
    
        print(f"Processing burst with RM = {rm}")
        print(f"Masking burst along phi between {phi_arr[mask_start]:.3f} and {phi_arr[mask_stop]:.3f}")
        print(f"Limiting SNR computations between {time[t1]:.3f} and {time[t2]:.3f} s")
    
        snr_t = []
    
        for t_idx in range(t1,t2):
            fdf = np.abs(data[t_idx, :])  # abs of FDF for this time bin
            peak_signal = fdf[rm_ind]  # signal at true rm
    
            # mask out region around the signal in phi
            mask = np.ones_like(fdf, dtype=bool)
            mask[max(0, mask_start):mask_stop] = False
            noise = fdf[mask]
    
            baseline = np.nanmedian(noise)  # absolute level
            sigma_noise = np.std(noise)  # std of noise
    
            # compute snr
            snr = (peak_signal - baseline) / sigma_noise
            snr_t.append(snr)
    
        # take max snr over time for this burst
        max_snr = np.max(snr_t)
        snr_max_per_peak.append(max_snr)
        print(f"Max SNR found: {max_snr:.3f}\n")

    return snr_max_per_peak



def plot_cpu_and_snr(param_arr, snr_arr, metadata_dict_list, timings_dict_list,
                     save_name="cpu_and_snr", save_loc=None, 
                     plot_ylog=True, t_true=[], rm_true=[]):
    
    # --- Info ---
    data_info = metadata_dict_list[0]
    t_true = data_info["sim_params"]['arrival_times']
    rm_true = data_info["sim_params"]['rm']

    # --- Colors ---
    rm_colors = ["#592e83", "#9984d4"]
    cpu_color = "#6a994e"
    
    # --- Figure Setup ---
    fig, ax1 = plt.subplots(figsize=(8,5))
    fig.suptitle(
        f"{data_info['fmin']:.1f} - {data_info['fmax']:.1f} MHz, "
        f"{data_info['delta_t_total_s']:.1f} s", y=0.95
    )
    
    # --- Left Axis: CPU Time ---
    cpu_data = [t_dict['total']['cpu_elapsed'] for t_dict in timings_dict_list]
    ax1.plot(param_arr, cpu_data, color=cpu_color, label="CPU time")
    ax1.set_xlabel("Downsampling Factor")
    ax1.set_ylabel("CPU Time [s]", color=cpu_color)
    ax1.tick_params(axis='y', labelcolor=cpu_color)
    
    if plot_ylog:
        ax1.set_yscale("log")
    
    # --- Right Axis: SNR ---
    ax2 = ax1.twinx()
    for i in range(len(t_true)):
        ax2.plot(
            param_arr,
            snr_arr[:, i],
            color=rm_colors[i],
            label=rf"$\phi$={rm_true[i]}, t={t_true[i]} s"
        )
    
    ax2.set_ylabel("S/N", color=rm_colors[0])
    ax2.tick_params(axis='y', labelcolor=rm_colors[0])
    
    
    # --- Legend ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False
    )
    
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    
    # --- Save Fig ---
    if save_name is not None:
        plt.savefig(save_loc + save_name, dpi=300)
    plt.close()