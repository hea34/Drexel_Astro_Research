import numpy as np
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.table import Table
from datetime import datetime
from astropy.time import Time
import re
import json

units = {
        'weeks': 604800,  # 60 * 60 * 24 * 7
        'days': 86400,    # 60 * 60 * 24
        'hours': 3600,    # 60 * 60
        'minutes': 60,
        'seconds': 1,
    }

def get_lightcurve(file, time_unit='d', threshold=1e3, std_devs=3, verbose=0, filename=None, filedir=".", filetype="lc"):
    """
    Helper function to retrieve timing info from raw BAT data.
    Combines observations across all energy channels and implements
    cuts to exclude outliers. Returns a dictionary with the full-band
    rates and errors, along with additional object and timing data.

    file (str): Name of FITS file containing Swift-BAT snapshot data.
    time_unit (str): Time unit of the returned light curve, one of
                     days ('d'), hours ('h'), minutes ('m'), or seconds ('s').
                     Default is 'd'. 
    threshold (int): Absolute value of the threshold used to determine "good"
                    data points (i.e. observations with a rate or error > +/- threshold 
                    are excluded)
    std_devs (int): Number of standard deviations used to define outliers. 
                    Defaults to 3.
    verbose (int): The verbosity level. Default is 0.
                    0: silent
                    1: basic (only prints the object name) 
                    2: normal (prints lightcurve info)
                    3: high (prints additional info about lightcurve corrections) 
                    4: debug (for debugging)
    filename (str): Name of file to save rebinned light curve to. If None, light curve won't be written to a file.
    filedir (str): Where to save the file if `filename` also supplied. Defaults to current working directory.
    filetype (lc): Type of file to save data to ('lc', 'csv', 'dat', 'parquet', 'txt', 'npz').
    
    returns: dict
        'time': mid-center time of observations 
                (in units of the input arg `time_unit`)
        'lumin': total rates from all energy bands
        'error': total propagated errors'start': start times of observations
        'end': end times of observations
        'exposure': exposure time of each observation
        'time_unit': unit of time sampling
        'MJDREF': MJD reference of light curve start
        'OBJECT': Counterpart name
        'BAT_NAME': Swift-BAT name
    """

    with fits.open(file) as hdu:

        # Primary HDU header
        hdr = hdu[1].header
        
        # Column names and data
        data = hdu[1].data
        cols = hdu[1].columns
        
        if verbose == 2:
            column_names = pd.DataFrame(cols.names, columns=['Column Names'])
            print(column_names.to_string(index=False))
        
        # Object information
        OBJECT = data['NAME'][0]
        if 'BAT_NAME' not in cols.names:
            BAT_NAME = np.nan
        else:
            BAT_NAME = data['BAT_NAME'][0]
            
        # Get timing info in seconds
        TSTART = data['TIME'][0]                                            # Starting time of file
        start_times = data['TIME'] - TSTART                                 # Start time of each observation (with respect to time = 0)
        exposure_times = np.array(data['EXPOSURE'], dtype=np.longdouble)    # Length of each observation, should in theory be the same as TSTOP - TSTART
        end_times = data['TIME'] + exposure_times - TSTART                  # End time of each observation (with respect to time = 0)
        center_times = start_times + (exposure_times)/2                     # Center time of each observation 
        
        if 'TIME_STOP' not in cols.names:   
            blank_sky_obj = False                                           # Not a blank sky object
            TSTOP = TSTART + data['EXPOSURE'][-1]                           # End time of file w/ respect to TIMESYS in seconds
            
            # if 'TIMEDEL' in cols.names:                                   # Not always included in fileTIMEDELTA = data['TIMEDEL'][-1]         
            #     # TIMEDELTA = data['TIMEDEL'][0]                          # If it is, delta time (length of light curve) is given in years
            #     continue
            # else: 
            TIMEDELTA = (TSTOP - TSTART)                                    # Convert to years for consistency with objects with data['TIMEDEL']
        else:
            blank_sky_obj = True                                            # Blank sky object
            TSTOP = data['TIME_STOP'][-1]                                   # Stopping time of last observation in seconds
            TIMEDELTA = (TSTOP - TSTART)                                    # Convert to years for consistency with non-blank sky objects

        if verbose == 3:                                                    # Print out times before converting units
            print("TSTART:", TSTART)
            print("TIME_STOP:", TSTOP)
            print("TIME_DELTA:", TIMEDELTA)
            
        elif verbose == 2:                                                    # Print additional info about corrections to light curve
                                                                            # (data cleaning procedures that were applied by the BAT team)
            print("\n\nCorrections applied:")
            print("\tGain correction:", data['GAINAPP'][0])
            print("\tBackground subtraction:", data['BACKAPP'][0])
            print("\tAutocollimation correction:", data['ACOLAPP'][0])
            print("\tPartial coding correction:", data['PCODEAPP'][0])
            print("\tProjection correction:", data['FFAPP'][0])
            print("\tBAT image distortion correction:", data['BDISTAPP'][0])
            print("\tMask weight technique correction:", data['MSKWTAPP'][0])
            print("\tOccultation correction:", data['OCCAPP'][0])
            
        # try: 
        #     MJDREF = hdr['MJDREFI'] + hdr['MJDREFF']            # Reference (start) time in MJD
        # except: 
            # datetime_string = search_FITS('datetime', hdu=hdu)  # Search header history for datetime string
            # MJDREF = Time(datetime.strptime(datetime_string, '%Y-%m-%dT%H:%M:%S'), format='datetime', scale='utc').mjd
            # if MJDREF and verbose == 4: print("Found a DateTime reference in FITS header: %s\n\tConverted to MJD -> %s" % (datetime_string, MJDREF))
            # if not MJDREF and verbose == 4: print("MJDREFI and MJDREFF not found in header... skipping")
            
        # Now need to combine the 8 energy channels into a summed light curve
        # and introduce some cuts for the rates and the errors to help exclude NaNs or ridiculous outliers

        # Get RATE and RATE_ERR arrays from data
        rates = np.array(data['RATE'], dtype=np.longdouble)
        errors = np.array(data['RATE_ERR'], dtype=np.longdouble)

        # Mask values outside the range [-threshold, threshold] with NaN
        rates[(rates < -1 * threshold) | (rates > threshold)] = np.nan
        errors[(errors < -1 * threshold) | (errors > threshold)] = np.nan

        # Square the errors for each band (error propagation means adding errors in quadrature)
        squared_errors = np.square(errors)

        # Sum across bands
        total_rates = np.nansum(rates, axis=1)
        total_squared_errors = np.nansum(squared_errors, axis=1)

        # Take square root of the total squared errors
        total_errors = np.sqrt(total_squared_errors)

        # Again require values to be inside the range [-1e4, 1e4]
        # total_rates[(total_rates < -1 * threshold) | (total_rates > 1 * threshold)] = np.nan
        # total_errors[(total_errors < -1 * threshold) | (total_errors > 1 * threshold)] = np.nan
        
        # Now exclude observations where error > N * std deviation of all errors
        # where N = `std_devs`
        large_errors = (total_errors > std_devs * np.nanstd(total_errors))
        total_errors[large_errors] = np.nan

        # Then exclude points with rate > larger than the mean +/- N std deviations of non-outliers
        good_rates = total_rates[~large_errors]
        mean_good = np.nanmean(good_rates)
        std_good = np.nanstd(good_rates)
        upper_limit = mean_good + std_devs * std_good
        lower_limit = mean_good - std_devs * std_good
        large_rates = ((good_rates > upper_limit) | (good_rates < lower_limit))
        good_rates[large_rates] = np.nan
        
        # Exclude the outliers by keeping non-NaN indices 
        mask = ~np.isnan(total_rates) & ~np.isnan(total_errors)
        start_times = start_times[mask]
        end_times = end_times[mask]
        exposure_times = exposure_times[mask]
        center_times = center_times[mask]
        total_rates = total_rates[mask]
        total_errors = total_errors[mask]

        # Convert time units since TIMESYS is in seconds (i.e. convert to 'time_unit' if not in seconds)
        units = {
            'weeks': 604800,  # 60 * 60 * 24 * 7
            'days': 86400,    # 60 * 60 * 24
            'hours': 3600,    # 60 * 60
            'minutes': 60,
            'seconds': 1,
        }
        
        if time_unit == 'd':                                   # Convert to days
            TSTART /= units['days']             
            start_times /= units['days']     
            exposure_times /= units['days']      
            end_times /= units['days']
            center_times /= units['days']
            TSTOP /= units['days'] 
            TIMEDELTA /= units['days']          

        elif time_unit == 'h':                                  # Convert to hours
            TSTART /= units['hours']                   
            start_times /= units['hours'] 
            exposure_times /= units['hours'] 
            end_times /= units['hours']
            center_times /= units['hours']
            TSTOP = TSTOP / units['hours']  
            TIMEDELTA /= units['hours']  
            
        if verbose == 3:                                        # Debug converted times and combined data
            print("\nTSTART:", TSTART)
            print("TIME_STOP:", TSTOP)
            print("TIME_DELTA:", TIMEDELTA)
            
            print("\nStart times:", start_times[:5])
            print("End times:", end_times[:5])
            print("Center times:", center_times[:5])
            print("Exposure times:", exposure_times[:5])
            
            print("\nRate limits: ", np.nanmin(rates), np.nanmax(rates))
            print("Error limits: ", np.nanmin(errors), np.nanmax(errors))
            
            print("\nTotal rate limits: ", np.nanmin(total_rates), np.nanmax(total_rates))
            print("Total error limits: ", np.nanmin(total_errors), np.nanmax(total_errors))

        elif verbose == 1:
            print("\n\nCombined light curve data:")
            # if MJDREF: print("\tMJD REF:", MJDREF)
            print('\tLength of light curve: %.2f years' % TIMEDELTA)
            print('\tRate [cts/s/pixel]:', total_rates)
            print('\tRate error [cts/s/pixel]:', total_errors)
            print('\tMean flux:', np.nanmean(total_rates))
            print('\tMean error:', np.nanmean(total_errors))
            
        lightcurve = {
            'time': center_times,
            'lumin': total_rates,
            'error': total_errors,
            'start': start_times,
            'end': end_times,
            'exposure': exposure_times,
            'time_unit': time_unit,
            # 'MJDREF': MJDREF if MJDREF else np.nan,
            'OBJECT': OBJECT, 
            'BAT_NAME': BAT_NAME,
        }
        
        df = pd.DataFrame(lightcurve)
        
        # Save rebinned light curve to file if filename supplied 
        if filename: 
            if filetype in ['lc', 'dat', 'txt', 'csv']: df.to_csv('%s/%s.%s' % (filedir, filename, filetype), index=None)
            elif filetype == 'parquet': df.to_parquet('%s/%s.%s' % (filedir, filename, filetype), index=None)
            elif filetype == 'npz': 
                np.savez('%s/%s.%s' % (filedir, filename, filetype), **lightcurve)
    
        return lightcurve

def rebin_light_curve(data, interval='1d', interpolate=False, fill_error='std', fraction=0.2, keep_empty=False, dataframe=True, filename=None, filedir=".", filetype="lc"):
    """
    Advanced resampling of light curve data using Pandas' resampling capabilities. If an interpolation
    scheme is specified, also performs interpolation of empty bins after resampling and computes errors
    on interpolated values. Data is assumed to be unbinned if keys starting with 'unbinned' aren't found
    in the supplied dictionary. Rebinned light curve is also written to a file if a filename is given.
    
    Parameters:
    data (dict): Dictionary containing 'time', 'lumin', and 'error' keys at a minimum.
    interval (str): Resampling interval, e.g., '5d', '1h', etc.
    interpolate (str or bool): Interpolation method ('slinear', 'linear', 'cubic', 'quadratic').
                               If True, use default method of linear spline ('slinear'). 
                               Defaults to False (no interpolation). 
    fill_error (str): Method to calculate errors on interpolated values ('std' deviation of observed values or
                      'mean' or 'median' of observed errors).
    fraction (float): Minimum fraction of bin width that must be covered by actual exposure.
    keep_empty (bool): Whether empty bins should be kept after resampling. If False, drop all empty (np.nan) bins. 
    dataframe (bool): Whether to return rebinned data as arrays or a Pandas DataFrame.
    filename (str): Name of file to save rebinned light curve to. If None, light curve won't be written to a file.
    filedir (str): Where to save the file if `filename` also supplied. Defaults to current working directory.
    filetype (lc): Type of file to save Pandas DataFrame as ('lc', 'csv', 'dat', 'parquet', 'txt', 'npz').
    
    Returns:
    dict (and pd.DataFrame): Resampled light curve with interpolation if specified. 'time', 'lumin', 'error', and
                             'exposure', keys of input `data` are modified in place, and unbinned values are saved to
                             'unbinned_time', 'unbinned_lumin', 'unbinned_error', and 'unbinned_exposure' keys.
    """
         
    # Error handling for input data
    if not isinstance(int(interval[0]), int) or not interval[1] in ['d', 'h', 'm', 'T', 's']:
        raise ValueError("Error: Invalid sampling frequency. `interval` should look like e.g. '1d'")
    
    is_unbinned = set(['time', 'lumin', 'error']).issubset(set(data.keys()))
    unbinned_keys = ['unbinned_time', 'unbinned_lumin', 'unbinned_error', 'unbinned_exposure']
    is_binned = set(unbinned_keys).issubset(set(data.keys()))
     
    if (not is_unbinned and not is_binned) or not isinstance(data, dict):
        raise ValueError("Error: Invalid input data. Must be a dictionary with at least 'time', 'lumin', 'error', and 'exposure' keys.")
    
    # Make sure any NumPy arrays are little-endian 
    for k, v in data.items(): 
        if isinstance(v, np.ndarray) and v.dtype.byteorder == '>':                                                        # '>' denotes big-endian
            print("key=", k, "\n value=", v, "byteorder=", v.dtype.byteorder, "dtype", v.dtype)
            v = v.byteswap().newbyteorder()                                              # Convert to little-endian to make Pandas happy

    # Create initial DataFrame 
    if is_binned:                                                                           # Use unbinned values if found
        unbinned_data = {new_key: data[key] for key, new_key in zip(unbinned_keys, ['time', 'lumin', 'error', 'exposure'])}
        df = pd.DataFrame(unbinned_data)
    elif is_unbinned:                                                                        # Assume 'time', 'lumin', and 'error' are unbinned
        df = pd.DataFrame(data)
        
        # Save the original unbinned data to 'unbinned_time', 'unbinned_lumin', 'unbinned_error' 
        for new_key, key in zip(unbinned_keys, ['time', 'lumin', 'error', 'exposure']):
            data[new_key] = data[key]
    else:
        raise ValueError("Error: Missing input data. Dictionary should contain 'time', 'lumin', 'error', and 'exposure' keys.")

     # Create DateTime index from unbinned times (after converting them to the units of `interval`)
    df.set_index(pd.to_datetime(df['time'], unit=data.get('unbinned_time_unit', 'd')  ), inplace=True)         

    # Resample data
    resampled = df.resample(interval).agg({
        'lumin': 'sum',
        'error': lambda x: np.sqrt(np.sum(x**2)),
        'exposure': 'sum'
    })
        
    # Get center time of each bin 
    interval_seconds = pd.to_timedelta(interval).total_seconds()                        # Duration of each bin in seconds
    unit = interval[-1].lower()                                                         # Assuming 'd', 'h', 'm', or 's'
    if unit == 'd': dt = interval_seconds / units['days']
    elif unit == 'h': dt = interval_seconds / units['hours']
    elif unit == 'm' or unit == 't':  
        unit = 'T'                                                                      # Convert 'm' to 'T' for minutes in Pandas
        dt = interval_seconds / units['minutes']
    else:                                                                               # Should just be 's' if here
        dt = interval_seconds
    resampled['time'] = np.arange(0, stop=len(resampled) * dt, step=dt)
     
    # Calculate total possible exposure per bin based on resampling frequency         
    resampled['obs_coverage'] = resampled['exposure'] / dt                  # Fraction of bin width covered by actual observations
    
    # Mask bins where exposure is less than specified `fraction`
    underexposed = resampled['obs_coverage'] < fraction
    resampled.loc[underexposed, ['lumin', 'error']] = np.nan

    # Interpolate empty bins (if interpolation method supplied)
    if interpolate:
        interpolation_method = interpolate if isinstance(interpolate, str) else 'slinear'
        resampled['interpolated'] = False                                   # Initialize column to track interpolated bins

        for col in ['lumin', 'error']:
            if not resampled[col].isna().all():                             # Only interpolate if there are non-NaN values
                # Detect where NaNs are before interpolation
                isna_before = resampled[col].isna()

                # Interpolate missing values
                valid_index = resampled[col].dropna().index
                interpolator = interp1d(valid_index, resampled.loc[valid_index, col], kind=interpolate,
                                        fill_value="extrapolate", bounds_error=False)
                resampled[col] = interpolator(resampled.index)

                # Detect where values were interpolated
                isna_after = resampled[col].notna() & isna_before

                # Update the interpolated count
                resampled.loc[isna_after, 'interpolated'] = True

        # Error handling on interpolated values
        if fill_error in ['std', 'mean', 'median']:
            if fill_error == 'std':                                                              # Calculate the standard deviation of the original, unbinned luminosity values
                std_dev = np.nanstd(df['lumin'])                                                 # df is the DataFrame containing original data
                resampled.loc[underexposed, 'error'] = std_dev
            else:                                                                                # Use the mean or median of the unbinned errors 
                error_stat = getattr(df['error'].dropna(), fill_error)()                        
                resampled.loc[underexposed, 'error'] = error_stat
      
    # Reset index for regular integer indexing and move 'time' to first column
    resampled.reset_index(drop=True, inplace=True)
    time_col = resampled.pop('time')
    resampled.insert(0, 'time', time_col)
    
    # Drop any empty bins unless keep_empty is True
    if not keep_empty:
        resampled = resampled.dropna()
        
    # Update original `data` dictionary with rebinned data and time_unit
    data['time'] = resampled['time'].values
    data['lumin'] = resampled['lumin'].values
    data['error'] = resampled['error'].values
    data['exposure'] = resampled['exposure'].values
        
    # Save old and new time unit
    data['unbinned_time_unit'] = data['time_unit']
    data['time_unit'] = unit
    
    # Save rebinned light curve to file if filename supplied 
    if filename: 
        if filetype in ['lc', 'dat', 'txt', 'csv']: resampled.to_csv('%s/%s.%s' % (filedir, filename, filetype), index=None)
        elif filetype == 'parquet': resampled.to_parquet('%s/%s.%s' % (filedir, filename, filetype), index=None)
        elif filetype == 'npz': 
            resampled_data = resampled.to_dict()
            np.savez('%s/%s.%s' % (filedir, filename, filetype), **resampled_data)
    
    # Return rebinned data
    if dataframe:
        return resampled
    else:
        return resampled.to_dict('list')

def plot_lightcurve(lc, color='r', marker='.', markersize=None, linestyle="", alpha=1, mask=None, errors=False, error_color=None, \
    error_alpha=0.3, capsize=0, ticks='all', figsize=None, margin=[0.01, 0.04], xlabel=True, ylabel=True, title=None, \
        filedir=".", filename=None, filetype='pdf', show=True):
    
    """ Plot lightcurve from data returned by get_lightcurve() function or a dictionary of arrays. 
    
    Parameters:
    lc (dict or DataFrame): Dictionary or DataFrame returned from get_lightcurve function. May also be a dictionary 
                            containing 'time' and 'flux' keys at a minimum.
    color (str): Color of scatter points and line, defaults to 'r' (red).
    marker (str): Scatter point marker style, default is '.' (point).
    markersize (int): Size of scatter points, default is 1 (or 3 if errors=True).
    linestyle (str): Style of the line connecting points, default is "" (no line).
    alpha (float): Opacity level of scatter points, default is 1 (opaque).
    mask (array-like, optional): Indices to mask, i.e., "bad" data points not excluded by get_lightcurve() or to reduce plot density.
    errors (bool): Whether to plot error bars, default is False.
    error_color (str): Color of error bars; defaults to None, which will use the value of `color`.
    error_alpha (float): Opacity level of error bars, default is 0.03 (translucent).
    capsize (int): Length of the caps on error bars, default is 0 (no caps).
    ticks (str, bool, or dict): Tick settings, defaults to 'all'. Can be None for matplotlib default, True for this function's default,
                                False for no ticks, 'all' for ticks on all sides with inward direction, or a dict for custom settings.
    figsize (tuple): Size of the plot, default is None which lets matplotlib choose.
    margin (list or tuple): Margins around the plot as a single float for both axes or a list [x_margin, y_margin], default is [0.01, 0.04].
    xlabel (str or bool): Label for the x-axis, default is True to set label as 't_rest (units)', where units are set to `lc['time_unit']` if found
                          and 'arb. units' otherwise. If False, axis isn't labeled.
    ylabel (str or bool): Label for the y-axis, default is True to set label as `Flux Rate (cts/s/pixel)`.
    title (str): Title of the figure, default is None (no title).
    filename (str): Name of the file to save the plot to. If None, the plot is not saved, default is None.
    filetype (str): File extension/type for saving the plot, choices include 'pdf', 'png', 'jpg', or 'jpeg', defaults to 'pdf'.
    show (bool): Whether to display the plot window, default is True.

    Returns:
    None
    """
    
    # check input data type
    # if isinstance(lc, pd.DataFrame): 
    
    # set up plot
    if not figsize: figsize = (10, 5)
    plt.figure(figsize=figsize) 
    
    # title and axis labels
    units = {'d': 'days', 's': 'seconds', 'h': 'hours', 'y': 'years', 'm': 'minutes'}
    if not xlabel: xlabel = ""
    elif xlabel and 'time_unit' in lc.keys(): xlabel = "$t_{rest}$ (%s)" % units[lc['time_unit']]
    elif xlabel and 'time_unit' not in lc.keys(): xlabel = "$t_{rest}$ (arb. units)"
    if isinstance(ylabel, bool) and not ylabel: ylabel = ""
    elif ylabel: ylabel = "Flux Rate (cts/s/pixel)"
    plt.ylabel(ylabel, fontsize=18, labelpad=6)
    plt.xlabel(xlabel, fontsize=18, labelpad=6)
    if title: plt.title(title, fontsize=18)
    
    # plot the light curve
    if errors: 
        if not markersize: markersize = 3
        if not error_color: error_color = color
        plt.errorbar(lc['time'], lc['lumin'], yerr=lc['error'], color=color, alpha=error_alpha, marker=marker, markersize=markersize, ecolor=error_color, capsize=capsize, linestyle=linestyle, zorder=1)
        plt.plot(lc['time'], lc['lumin'], linestyle=linestyle, linewidth=1, alpha=alpha, color=color, marker=marker, markersize=markersize, zorder=2)
    else:
        if not markersize: markersize = 1
        plt.plot(lc['time'], lc['lumin'], linestyle=linestyle, linewidth=1, alpha=alpha, color=color, marker=marker, markersize=markersize)
        
    # adjust tick settings
    if not ticks:                               # turn off ticks
        plt.tick_params(which='both', axis='both', right=False, top=False, bottom=False, left=False, labelleft=False, labelbottom=False)
    elif ticks:                                 # use this function's default styling
        plt.tick_params(which='both', axis='both', direction='out', right=False, top=False, bottom=True, left=True, labelleft=True, labelbottom=True, pad=8, labelsize=16, width=1)
        plt.tick_params(which='major', axis='both', length=10)
        plt.tick_params(which='minor', axis='both', length=5)
    elif ticks == 'all':                        # use default styling but add ticks on top and right facing 'in'
        plt.tick_params(which='both', axis='both', direction='in', right=True, top=True, bottom=True, left=True, labelleft=True, labelbottom=True)
    if ticks and isinstance(ticks, dict):       # custom tick settings
        plt.tick_params(**ticks)
    
    # adjust margins
    if margin and isinstance(margin, (int, float)):
        margin = {'x': margin, 'y': margin}
    if margin and isinstance(margin, (list, tuple, np.ndarray)):
         margin = {'x': margin[0], 'y': margin[0]}
    plt.margins(**margin)                  
    plt.gca().autoscale_view()
        
    # save and show
    if filetype not in ['pdf', 'png', 'jpeg', 'jpg']: 
        filetype = 'pdf'   # fallback to PDF 
        plt.savefig("%s/%s.%s" % (filedir, filename, filetype), bbox_inches='tight')
    if show: plt.show()
    
### HELPER FUNCTIONS ###

def time_to_MJD(times, MJDREF, time_unit):
            
    # Calculate relative time offsets (to t=0) in days
    if time_unit.lower() == 's':                                        # Convert seconds to days
        time_days = times / units['days']
    elif time_unit.lower() == 'h':
        time_days = (times * units['hours']) / units['days']            # Convert hours to seconds to days
    elif time_unit.lower() == 'm' | time_unit == 't':
        time_days = (times * units['minutes']) / units['days']          # Convert minutes to seconds to days
    elif time_unit.lower() == 'd':
        time_days = times                                               # Already in days
    else:
        raise ValueError("Error: Invalid time_unit. Must be one of\n\t's': seconds\n\t'm' or 't': minutes\n\t'h': hours\n\t'd': days")
    
    # Convert to MJD by adding the MJDREF
    if not isinstance(MJDREF, str): raise ValueError("Error: Invalid MJDREF ")
    return mjdref + time_days


def search_FITS(entry_type, file=None, hdu=None, search_string=None, pattern=None, verbose=2):
    """ Search history of FITS file for observation info based on specified entry_type. 
    
    Parameters:
    entry_type (str): Search term, expected to be a key in the mapping dictionary `entry_config`.
    file (str): Optional. Path to FITS file of light curve to search. 
    hdu (object): Optional. FITS data object containing the file header.
    search_string (str): Optional. Custom search string. 
    pattern (str): Optional. Regex matching pattern to extract a term from a string matching `search_string`.
                    If `search_string` is not None but no `pattern` is supplied, the extracted term will be 
                    the substring following the last space in the matching string.
                    
    Returns:
    match (str or array of str): the extracted substring(s)
    """

    if not file and not hdu: 
        raise ValueError("Error: Either 'file' path or open FITS 'hdu' object must be supplied.")
    
    # Mapping of some common entry types in header['HISTORY'] to their corresponding 
    # search strings and regex patterns
    entry_config = {
        'datetime': {
            'search_string': '',  # No specific context string required for general search
            'pattern': r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
        },
        'MJD': {
            'search_string': 'MJDREF',
            'pattern': r'MJDREF\s*=\s*(\d+\.\d+)'
        },
        'object': {
            'search_string': 'obj',
            'pattern': r'obj\s*=\s*(\w+)'
        },
        'name': {
            'search_string': 'name',
            'pattern': r'name\s*=\s*(\w+)'
        }
    }

    if entry_type not in entry_config:
        raise ValueError("Error: Invalid entry_type. Supported types are: 'datetime', 'MJD', 'object', 'name'.")

    # Get the search string and pattern for the specified entry_type if not supplied
    if search_string is None: search_string = entry_config[entry_type]['search_string']
    if pattern is None: pattern = entry_config[entry_type]['pattern']

    # Open the FITS file
    if hdu:         # If `hdu` or both supplied, use the open hdu (i.e. if calling within a function)
        header = hdu[1].header
    elif file:
        with fits.open(file) as hdul:
            # Access the primary header
            header = hdul[1].header
   
    # Loop through each entry in the HISTORY of the header
    for history_entry in header['HISTORY']:
        # Check if this history entry contains the specific search string
        if search_string in history_entry:
            # Use the regex to find the specified pattern in the entry
            match = re.search(pattern, history_entry)
            if match:
                return match.group(1) if len(match.groups()) > 0 else match.group()  # Return the matched string
                    
    return None  # Return None if no matching entry is found

class LightCurve:
    def __init__(self, file=None, instrument=None, t=None, y=None, yerr=None, object=None, MJD=None, config_file=None):
        """
        Initializes a new LightCurve object.

        Parameters:
        file (str): Optional. Path to the file containing light curve data.
        instrument (str): Optional. Name of the instrument used to collect the data.
        t (array-like): Optional. Times of the observations.
        y (array-like): Optional. Luminosity or flux measurements of the light curve.
        yerr (array-like): Optional. Errors associated with the luminosity or flux measurements.
        object (str): Optional. Name or identifier of the observed object.
        MJD (int): Optional. Reference MJD date for start of the light curve.
        config_file (str): Optional. Path to config.json file.
        """

        self.file = file
        self.instrument = instrument
        self.times = t
        self.lumin = y
        self.error = yerr
        self.object = object
        self.mjd_ref = MJD
        
        # Initialize apping of some common entry types in header['HISTORY'] to their corresponding 
        # search strings and regex patterns
        if config_file is not None:
            with open(config_file, 'r') as file:
                self.entry_config = json.load(file)
        else:
            entry_config = {
                'datetime': {
                    'search_string': '',  # No specific context string required for general search
                    'pattern': r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
                },
                'MJD': {
                    'search_string': 'MJDREF',
                    'pattern': r'MJDREF\s*=\s*(\d+\.\d+)'
                },
                'object': {
                    'search_string': 'obj',
                    'pattern': r'obj\s*=\s*(\w+)'
                },
                'name': {
                    'search_string': 'name',
                    'pattern': r'name\s*=\s*(\w+)'
                }
            }

    def load(self, filepath):
        """
        Load data from a file.
        
        Parameters:
        filepath (str): Path to the data file.
        """
        # raise NotImplementedError("This method needs to be implemented to read light curve data from a file.")

        # Search FITS header for a datetime string to set MJD reference to
        datetime_string = search_history(filepath, entry_type='datetime')
        self.set_mjd_ref(datetime_string)
        
        self.start_time = start_times
        self.end_time = end_times
        self.exposure_time = exposure_time
        self.TSTART = TSTART
        self.TSTOP = TSTOP
        self.lumin = lumin
        self.errors = errors
        
    def plot(self):
        """
        Plot the light curve.
        """
        raise NotImplementedError("This method needs to be implemented to plot the light curve.")

    def save(self, filename):
        """
        Save the light curve data to a file.
        
        Parameters:
        filename (str): The name of the file to save the data.
        """
        raise NotImplementedError("This method needs to be implemented to save the light curve data.")
    
    def to_df(self, filename):
        """
        Returns a Pandas DataFrame to a file.
        
        Parameters:
        filename (str): The name of the file to save the data.
        """
        raise NotImplementedError("This method needs to be implemented to save the light curve data.")
    
    def set_mjd_ref(self, datetime_string, time_scale='utc'):
        """ Converts a ISO 8601 datetime string to Julian Date and updates the object's MJD reference. 
        
        Parameters:
        datetime_string (str): The datetime string in ISO 8601 format.
        time_scale (str): The time scale of the datetime string, e.g. 'utc', 'tai', 'tt', etc. 
                          Defaults to UTC.
        """
        
        t = Time(datetime.strptime(datetime_string, '%Y-%m-%dT%H:%M:%S'), format='datetime', scale=time_scale)
        self.mjdref = t.mjd
    
    def search_FITS(self, entry_type, search_string=None, pattern=None, verbose=2):
        """ Search history of FITS file header for observation info based on specified entry_type. 
        
        Parameters:
        file (str): Path to FITS file of light curve to search. 
        entry_type
        """

        if entry_type not in self.search_config:
            raise ValueError("Error: Invalid entry_type. Supported types are: 'datetime', 'MJD', 'object', 'name'.")

        # Get the search string and pattern for the specified entry_type if not supplied
        if search_string is None: search_string = self.search_config[entry_type]['search_string']
        if pattern is None: pattern = self.search_config[entry_type]['pattern']

        # Open the FITS file
        with fits.open(self.file) as hdul:
            # Access the primary header
            header = hdul[1].header
            
            # Loop through each entry in the HISTORY of the header
            for history_entry in header['HISTORY']:
                # Check if this history entry contains the specific search string
                if search_string in history_entry:
                    # Use the regex to find the specified pattern in the entry
                    match = re.search(pattern, history_entry)
                    if match:
                        return match.group(1) if len(match.groups()) > 0 else match.group()  # Return the matched string
                        
        return None  # Return None if no matching entry is found

    def print_search_terms(self):
        print("Terms to search FITS headers for:\n")
        for term, details in self.search_config.items():
            print(f" {term} ")
            print(f"\t\t\t- search string: \"{details['search_string']}\"")
            print(f"\t\t\t - pattern: \"{details['pattern']}\"\n")

    def update_search_term(self, entry_term, search_string=None, pattern=None):
        if entry_term.lower() in self.search_config:
            if search_string:
                self.search_config[entry_term]['search_string'] = search_string
            if pattern:
                self.search_config[entry_term]['pattern'] = pattern
            self._update_config_file()
        else:
            print(f"No such term '{entry_term}' in the configuration.")

    def add_search_term(self, entry_type, search_string, pattern=None):
        if pattern is None:
            pattern = '\\S+$'
        self.entry_config[entry_type] = {
            'search_string': search_string,
            'pattern': pattern
        }
        self._update_config_file()

    def _update_config_file(self, config_path='config.json'):
        with open(config_path, 'w') as file:
            json.dump(self.entry_config, file, indent=4)
            
    def rebin(self, interval='1d', interpolate='slinear', fill_error='std', fraction=0.02, dataframe=False):
        """
        Advanced rebining of light curve data using Pandas' resampling capabilities.

        Parameters:
        data (dict): Dictionary containing 'start_time', 'exposure', 'end_time', 'lumin', and 'error'.
        interval (str): Resampling interval, e.g., '5d', '1h', etc.
        interpolate (str or None): Interpolation method ('slinear', 'linear', 'cubic', 'quadratic', None).
        fill_error (str): Method to fill errors on interpolated values ('std', 'mean', 'median').
        fraction (float): Minimum fraction of total bin interval that must be covered by actual exposure.
        dataframe (bool): Whether to return rebinned data as arrays or a Pandas DataFrame.
        
        Returns:
        pd.DataFrame: DataFrame with resampled and interpolated 'time', 'lumin', and 'error'.
        """
        
        # Format `interval` string and get unit from last character
        interval = interval.lower().replace(" ", "")
        unit = interval[-1].upper()
        
        # Create initial DataFrame with unbinned data
        data = {'start_time': self.start_time, 'exposure': self.exposure_time, 
                'end_time': self.end_time, 'lumin': self.lumin, 'error': self.error}
        df = pd.DataFrame(data)
        
        # Calculate center time of each observation
        df['time'] = df['start_time'] + df['exposure'] / 2
        df.set_index(pd.to_datetime(df['time'], unit=unit), inplace=True)

        # Resample data
        resampled = df.resample(interval).agg({
            'lumin': 'sum',
            'error': lambda x: np.sqrt(np.sum(x**2)),
            'exposure': 'sum'
        })

        # Calculate total possible exposure per bin based on resampling frequency
        resampled['expected_exposure'] = pd.to_timedelta(interval).total_seconds()
        resampled['coverage'] = resampled['exposure'] / resampled['expected_exposure']

        # Fill bins where exposure is less than specified fraction
        underexposed = resampled['coverage'] < fraction
        resampled.loc[underexposed, 'lumin'] = np.nan
        resampled.loc[underexposed, 'error'] = np.nan

        # Interpolation (if specified)
        if interpolate:
            for col in ['lumin', 'error']:
                if not resampled[col].isna().all():  # Only interpolate if there are non-NaN values
                    f = interp1d(resampled.dropna().index.values, resampled.dropna()[col], kind=interpolate, fill_value="extrapolate")
                    resampled[col] = f(resampled.index.values)

        # Error handling on interpolated values
        if fill_error in ['std', 'mean', 'median']:
            if fill_error == 'std':
                # Calculate the standard deviation of the original, unbinned luminosity values
                std_dev = np.std(df['lumin'])  # Assuming df is the DataFrame containing original data
                resampled.loc[underexposed, 'error'] = std_dev
            else:
                # Use the mean or median of the rebinned errors
                error_stat = getattr(resampled['error'], fill_error)()  # Get mean/median of rebinned errors
                resampled.loc[underexposed, 'error'] = error_stat
        else: 
            raise ValueError("Error: Invalid method for calculating errors on interpolated bins. \
                             \nSupported types are:\n\t'std': standard deviation of observed (unbinned) rates\
                            \n\t'mean': average observed error \n\t'median': median observed error.")

        # Update attributes with the rebinned data
        self.t = resampled.index.to_julian_date() - self.mjdref
        self.y = resampled['lumin'].values
        self.yerr = resampled['error'].values

        # Save for later access to the DataFrame
        self.rebinned_df = resampled.reset_index()
        
        return resampled.reset_index()

# Example usage:
# Creating an instance without any data.
lc_empty = LightCurve()

# Creating an instance with some data.
lc_with_data = LightCurve(t=[0, 1, 2, 3], y=[10, 15, 12, 14], yerr=[1, 0.5, 1.2, 0.8], object="Star XYZ")

# These instances would now have the appropriate properties set, and the placeholder methods are ready to be implemented.
