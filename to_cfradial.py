#!/usr/bin/env python3
# Convert OU's netcdf radar files to cfradial format
# Created 29 Janurary 2024 by Sam Gardner <samuel.gardner@ttu.edu>

from os import path, listdir
from sys import argv
import xarray as xr
import numpy as np
import pyart
from datetime import datetime as dt, UTC
import tarfile
from io import BytesIO 



def make_cfradial_format_from_radars(datasets, time_spacing_to_next_scan):
    files_to_combine = []
    fields = dict()
    for ds in datasets:
        datafilled = ds[ds.attrs['TypeName']].data
        datafilled[np.where(datafilled == ds.attrs['MissingData'])] = -9999
        datafilled[np.where(datafilled == ds.attrs['RangeFolded'])] = -9999
        match  ds.attrs['TypeName']:
            case 'Intensity':
                fields['reflectivity'] = {
                    'units' : ds.attrs['Unit-value'],
                    'standard_name' : 'equivalent_reflectivity_factor',
                    'long_name' : 'Reflectivity',
                    'coordinates' : 'elevation azimuth range',
                    '_FillValue' : -9999,
                    'data' : datafilled
                }
            case 'Radial_Velocity':
                fields['velocity'] = {
                    'units' : ds.attrs['Unit-value'],
                    'standard_name' : 'radial_velocity_of_scatterers_away_from_instrument',
                    'long_name' : 'Mean doppler Velocity',
                    'valid_max' : ds.attrs['Nyquist_Vel-value'],
                    'valid_min' : -ds.attrs['Nyquist_Vel-value'],
                    'coordiantes' : 'elevation azimuth range',
                    '_FillValue' : -9999,
                    'data' : datafilled
                }
            case 'Width':
                fields['spectrum_width'] = {
                    'units' : ds.attrs['Unit-value'],
                    'standard_name' : 'doppler_spectrum_width',
                    'long_name' : 'Spectrum Width',
                    'valid_max' : 2*ds.attrs['Nyquist_Vel-value'],
                    'valid_min' : 0.0,
                    'coordinates' : 'elevation azimuth range',
                    '_FillValue' : -9999,
                    'data' : datafilled
                }
            case 'Differential_Reflectivity':
                fields['differential_reflectivity'] = {
                    'units' : ds.attrs['Unit-value'],
                    'standard_name' : 'log_differential_reflectivity_hv',
                    'long_name' : 'log_differential_reflectivity_hv',
                    'coordinates' : 'elevation azimuth range',
                    '_FillValue' : -9999,
                    'data' : datafilled
                }
            case 'PhiDP':
                fields['differential_phase'] = {
                    'units' : ds.attrs['Unit-value'],
                    'standard_name' : 'differential_phase_hv',
                    'long_name' : 'differential_phase_hv',
                    'valid_max' : np.pi,
                    'valid_min' : -np.pi,
                    'coordinates' : 'elevation azimuth range',
                    '_FillValue' : -9999,
                    'data' : datafilled
                }
            case 'RhoHV':
                fields['cross_correlation_ratio'] = {
                    'units' : ds.attrs['Unit-value'],
                    'standard_name' : 'cross_correlation_ratio_hv',
                    'long_name' : 'cross_correlation_ratio_hv',
                    'valid_max' : 1.0,
                    'valid_min' : 0.0,
                    'coordinates' : 'elevation azimuth range',
                    '_FillValue' : -9999,
                    'data' : datafilled
                }
            case 'KDP':
                if ds.attrs['Unit-value'] == 'DegreesPerMeter':
                    datafilled = datafilled / 1000
                    this_unit = 'degrees/km'
                else:
                    this_unit = ds.attrs['Unit-value']
                fields['specific_differential_phase'] = {
                    'units' : this_unit,
                    'standard_name' : 'specific_differential_phase_hv',
                    'long_name' : 'Specific differential phase (KDP)',
                    'coordinates' : 'elevation azimuth range',
                    '_FillValue' : -9999,
                    'data' : datafilled
                }
            case 'Co-cross_Differential_Phase_H':
                fields['cross_polar_differential_phase'] = {
                    'units' : ds.attrs['Unit-value'],
                    'standard_name' : 'cross_polar_differential_phase_h',
                    'long_name' : 'cross_polar_differential_phase_h',
                    'valid_max' : np.pi,
                    'valid_min' : -np.pi,
                    'coordinates' : 'elevation azimuth range',
                    '_FillValue' : -9999,
                    'data' : datafilled
                }
            case 'Co-cross_Correlation_Coefficient_H':
                fields['co_to_cross_polar_correlation_ratio_h'] = {
                    'units' : ds.attrs['Unit-value'],
                    'standard_name' : 'co_to_cross_polar_correlation_ratio_h',
                    'long_name' : 'co_to_cross_polar_correlation_ratio_h',
                    'valid_max' : 1.0,
                    'valid_min' : 0.0,
                    'coordinates' : 'elevation azimuth range',
                    '_FillValue' : -9999,
                    'data' : datafilled
                }
            case 'Co-cross_Correlation_Coefficient_V':
                fields['co_to_cross_polar_correlation_ratio_v'] = {
                    'units' : ds.attrs['Unit-value'],
                    'standard_name' : 'co_to_cross_polar_correlation_ratio_v',
                    'long_name' : 'co_to_cross_polar_correlation_ratio_v',
                    'valid_max' : 1.0,
                    'valid_min' : 0.0,
                    'coordinates' : 'elevation azimuth range',
                    '_FillValue' : -9999,
                    'data' : datafilled
                }
            case 'Linear_Depolarization_Ratio_H':
                fields['linear_depolarization_ratio_h'] = {
                    'units' : ds.attrs['Unit-value'],
                    'standard_name' : 'log_linear_depolarization_ratio_h',
                    'long_name' : 'Linear depolarization ratio horizontal',
                    'coordinates' : 'elevation azimuth range',
                    '_FillValue' : -9999,
                    'data' : datafilled
                }
            case 'Linear_Depolarization_Ratio_V':
                fields['linear_depolarization_ratio_v'] = {
                    'units' : ds.attrs['Unit-value'],
                    'standard_name' : 'log_linear_depolarization_ratio_v',
                    'long_name' : 'Linear depolarization ratio vertical',
                    'coordinates' : 'elevation azimuth range',
                    '_FillValue' : -9999,
                    'data' : datafilled
                }
            
        files_to_combine.append(ds)
    time_start = dt.fromtimestamp(ds.attrs['Time'], UTC)
    num_rays = ds[list(ds.coords)[0]].data.shape[0]
    times_at_rays = np.linspace(ds.attrs['Time'], ds.attrs['Time'] + time_spacing_to_next_scan, num_rays)
    times_at_rays = np.array(times_at_rays)
    
    time_dict = {
        'units' : f'seconds since {time_start.strftime("%Y-%m-%dT%H:%M:%SZ")}',
        'standard_name' : 'time',
        'long_name' : 'time_in_seconds_since_volume_start',
        'calendar' : 'gregorian',
        'comment' : 'Coordinate variable for time. Time at the center of each ray, in fractional seconds since the global variable time_coverage_start',
        'data' : times_at_rays - ds.attrs['Time']
    }
    rng = np.matmul(ds.Gate.data.reshape(-1, 1), ds.GateWidth.data.reshape(1, -1))+ ds.attrs['RangeToFirstGate'] + np.median(ds.GateWidth.data)/2
    range_dict = {
        'units' : 'meters',
        'standard_name' : 'projection_range_coordinate',
        'long_name' : 'range_to_measurement_volume',
        'axis': 'radial_range_coordinate',
        'spacing_is_constant' : 'true',
        'comment' : 'Coordinate variable for range. Range to center of each bin.',
        'data' : rng[:, 0]
    }

    metadata_dict = {
        'Conventions' : 'CF/Radial instrument_parameters',
        'version' : '1.3',
        'title' : '',
        'institution' : 'OU ARRC',
        'references' : '',
        'source' : '',
        'history' : '',
        'comment' : '',
        'instrument_name' : '',
    }
    az = np.tile(ds.Azimuth.data, (rng.shape[0], 1))
    az_dict = {
        'units' : 'degrees',
        'standard_name' : 'beam_azimuth_angle',
        'long_name' : 'azimuth_angle_from_true_north',
        'axis' : 'radial_azimuth_coordinate',
        'comment' : 'Azimuth of antenna relative to true north',
        'data' : az[0, :]
    }

    eldat = np.tile(ds.Elevation.data, (rng.shape[0], 1))
    el_dict = {
        'units' : 'degrees',
        'standard_name' : 'beam_elevation_angle',
        'long_name' : 'elevation_angle_from_horizontal_plane',
        'axis' : 'radial_elevation_coordinate',
        'comment' : 'Elevation of antenna relative to the horizontal plane',
        'data' : eldat[0, :]
    }


    lat_dict = {
        'long_name' : 'Latitude',
        'standard_name' : 'Latitude',
        'units' : 'degrees_north',
        'data' : np.array([ds.attrs['LatitudeDouble']])
    }
    lon_dict = {
        'long_name' : 'Longitude',
        'standard_name' : 'Longitude',
        'units' : 'degrees_east',
        'data' : np.array([ds.attrs['LongitudeDouble']])
    }

    alt_dict = {
        'long_name' : 'Altitude',
        'standard_name' : 'Altitude',
        'units' : 'meters',
        'positive' : 'up',
        'data' : np.array([-9999])
    }



    sweep_dict = {
        'units' : 'count',
        'standard_name' : 'sweep_number',
        'long_name' : 'Sweep Number',
        'data' : np.array([0])
    }

    sweepmode_dict = {
        'units': 'unitless',
        'standard_name': 'sweep_mode',
        'long_name': 'Sweep mode',
        'comment': 'Options are: "sector", "coplane", "rhi", "vertical_pointing", "idle", "azimuth_surveillance", "elevation_surveillance", "sunscan", "pointing", "manual_ppi", "manual_rhi"',
    }

    fixangle_dict = {
        'long_name' : 'Target angle for sweep',
        'units' : 'degrees',
        'standard_name' : 'target_fixed_angle'
    }

    if ds.attrs['ScanType'] == 'RHI':
        sweepmode_dict['data'] = np.array([b'manual_rhi'])
        fixangle_dict['data'] = np.array([np.mean(az)])
    elif ds.attrs['ScanType'] == 'PPI':
        sweepmode_dict['data'] = np.array([b'manual_ppi'])
        fixangle_dict['data'] = np.array([np.mean(eldat)])

    startray_dict = {
        'long_name': 'Index of first ray in sweep, 0-based',
        'units': 'count',
        'data': np.array([0])
        }
    endray_dict = {
        'long_name': 'Index of last ray in sweep, 0-based',
        'units': 'count',
        'data': np.array([num_rays-1])
    }
    radar = pyart.core.radar.Radar(time_dict, range_dict, fields, metadata_dict, ds.attrs['ScanType'], lat_dict, lon_dict, alt_dict, sweep_dict, sweepmode_dict, fixangle_dict, startray_dict, endray_dict, az_dict, el_dict)
    return radar



if __name__ == '__main__':
    if len(argv) < 2:
        print('Usage: to_cfradial.py <path to .tar.xz files>')
        exit()
    in_dir = argv[1]
    if path.exists(in_dir) and path.isdir(in_dir):
        infiles = sorted(listdir(in_dir))
        times_of_scans = []
        orig_filenames = []
        for infile in infiles:
            if infile.endswith('.tar.xz'):
                time_of_scan = dt.strptime(infile.split('-')[1]+infile.split('-')[2], '%Y%m%d%H%M%S')
                times_of_scans.append(time_of_scan)
        for i in range(1, len(times_of_scans)):
            infile = infiles[i-1]
            this_sweep_time = times_of_scans[i-1]
            next_sweep_time = times_of_scans[i]
            time_spacing = (next_sweep_time - this_sweep_time).total_seconds()
            if infile.endswith('.tar.xz'):
                datasets = []
                with tarfile.open(path.join(in_dir, infile), 'r:xz') as tar:
                    for member in tar.getmembers():
                        if member.name.endswith('.nc'):
                            thisFile = tar.extractfile(member)
                            thisFileBytes = BytesIO(thisFile.read())
                            thisFile.close()
                            thisDataset = xr.open_dataset(thisFileBytes)
                            datasets.append(thisDataset)
                radar = make_cfradial_format_from_radars(datasets, time_spacing)
                pyart.io.write_cfradial(path.join(in_dir, infile.replace('.tar.xz', '.nc')), radar)
    else:
        print('Error: input directory does not exist')
        exit()