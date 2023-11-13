from pyxlma.lmalib.io import read as lma_read
from pyxlma.coords import RadarCoordinateSystem, TangentPlaneCartesianSystem
from pyxlma.lmalib.flash.cluster import cluster_flashes
from pyxlma.lmalib.flash.properties import flash_stats
from pyxlma.plot.xlma_base_plot import BlankPlot
from pyxlma.plot.xlma_plot_feature import color_by_time,  plot_points, subset
import numpy as np
from os import path, listdir
import pandas as pd
from cartopy import crs as ccrs
from pathlib import Path
from datetime import datetime as dt
import xarray as xr
from matplotlib import pyplot as plt
from cartopy import feature as cfeat
from metpy.plots import USCOUNTIES
import pyart


if __name__ == '__main__':
    caseType = 'iphone' # 'iphone' 'highspeed'
    if caseType == 'iphone':
        cases = [
            {
                'flid' : 12650,
                'deltalat' : 0.5,
                'deltalon' : 0.5,
                'lat_offset' : 0,
                'lon_offset' : 0,
                'start_offset' : -0.1e9,
                'end_offset' : 0.1e9,
                'px1k-dumb': [0, 0]
            }, {
                'flid' : 13367,
                'deltalat' : 0.2,
                'deltalon' : 0.2,
                'lat_offset' : 0,
                'lon_offset' : 0,
                'start_offset' : -0.04e9,
                'end_offset' : 0.006e9,
                'px1k-dumb': [0, 0]
            }, {
                'flid' : 14134,
                'deltalat' : 0.2,
                'deltalon' : 0.2,
                'lat_offset' : 0,
                'lon_offset' : 0,
                'start_offset' : -0.1e9,
                'end_offset' : 0,
                'px1k-dumb': [0, 0]
            }, {
                'flid' : 22059,
                'deltalat' : 0.25,
                'deltalon' : 0.25,
                'lat_offset' : 0,
                'lon_offset' : 0,
                'start_offset' : -0.02e9,
                'end_offset' : 0.25e9,
                'px1k-dumb': [-3, 3]
            }, {
                'flid' : 36458,
                'deltalat' : 0.5,
                'deltalon' : 0.5,
                'lat_offset' : 0,
                'lon_offset' : 0,
                'start_offset' : -0.02e9,
                'end_offset' : 0.55e9,
                'px1k-dumb': [0, 1]
            }, {
                'flid' : 41141,
                'deltalat' : 0.25,
                'deltalon' : 0.25,
                'lat_offset' : 0,
                'lon_offset' : 0,
                'start_offset' : -0.025e9,
                'end_offset' : 0.31e9
            }
        ]
        lmad, file_start = lma_read.dataset(['lma/0616/LYLOUT_230616_010000_0600.dat.gz'])
    else:
        cases = [
            {
                'flid' : 18439,
                'deltalat' : 0.25,
                'deltalon' : 0.25,
                'lat_offset' : 0,
                'lon_offset' : 0,
                'start_offset' : -0.15e9,
                'end_offset' : 0
            }, {
                'flid' : 28439,
                'deltalat' : 0.2,
                'deltalon' : 0.2,
                'lat_offset' : 0,
                'lon_offset' : 0,
                'start_offset' : -0.4e9,
                'end_offset' : 0
            }
        ]
        lmad, file_start = lma_read.dataset(['lma/0616/LYLOUT_230616_013000_0600.dat.gz'])
    radar_fields = [
        {
        'suffix' : 'Z',
        'varname' : 'Intensity',
        'label' : r'$Z$ dBZ',
        'cmap' : 'pyart_ChaseSpectral',
        'vmin' : -10,
        'vmax' : 80
        }, {
        'suffix' : 'D',
        'varname' : 'Differential_Reflectivity',
        'label' : r'$Z_{DR}$ dB',
        'cmap' : 'pyart_turbone_zdr',
        'vmin' : -2,
        'vmax' : 8
        }, {
        'suffix' : 'R',
        'varname' : 'RhoHV',
        'label' : r'$\rho_{HV}$',
        'cmap' : 'pyart_plasmidis',
        'vmin' : 0,
        'vmax' : 1
        }
    ]
    px = 1/plt.rcParams['figure.dpi']
    rax_times = {f'{el:.1f}' : [dt.strptime('_'.join(file.split('-')[1:3]), '%Y%m%d_%H%M%S') for file in sorted(listdir('rax')) if '-Z.nc' in file and f'-E{el:.1f}' in file] for el in np.arange(1, 30, 1.5)}
    rax_elevation_m = 380

    px_times = [dt.strptime('_'.join(file.split('-')[1:3]), '%Y%m%d_%H%M%S') for file in sorted(listdir('px1k')) if '-Z.nc' in file]
    px_elevation_m = 318

    lmad = flash_stats(cluster_flashes(lmad))

    basepath = path.dirname(path.realpath(__file__))
    outbasepath = path.join(basepath, 'polarimetry-'+caseType)

    for i in range(len(cases)):
        case = cases[i]
        target_flash = lmad.sel(number_of_flashes=case['flid'])
        target_lat = target_flash.flash_center_latitude.data + case['lat_offset']
        target_lon = target_flash.flash_center_longitude.data + case['lon_offset']
        time_start = np.array([target_flash.flash_time_start.data.astype(float) + case['start_offset'] ]).astype('datetime64[ns]')[0]
        time_start_dt = pd.Timestamp(time_start).to_pydatetime()
        time_end = np.array([target_flash.flash_time_end.data.astype(float) + case['end_offset'] ]).astype('datetime64[ns]')[0]
        time_end_dt = pd.Timestamp(time_end).to_pydatetime()

        lon_range = (target_lon - case['deltalon'], target_lon + case['deltalon'])
        lat_range = (target_lat - case['deltalat'], target_lat + case['deltalat'])
        alt_range = (0, 20)
        time_range = (time_start, time_end)

        lonSet, latSet, altSet, timeSet, selectedData = subset(lmad.event_longitude.data, lmad.event_latitude.data, lmad.event_altitude.data/1000, pd.Series(lmad.event_time), lmad.event_chi2.data, lmad.event_stations.data, lon_range, lat_range, alt_range, time_range, 1, 6)
        
        vmin, vmax, relative_colors = color_by_time(timeSet, [time_start_dt, time_end_dt])

        lmaPlot = BlankPlot(time_start_dt, bkgmap=True, xlim=lon_range, ylim=lat_range, zlim=alt_range, tlim=[time_start_dt, time_end_dt], title='OkLMA VHF Sources')
        
        lmaPlot.ax_plan.scatter(lmad.station_longitude, lmad.station_latitude, 20, 'white', 'D', linewidths=.5, edgecolors='k', transform=ccrs.PlateCarree(), zorder=4)
        
        camera_lon = -97.581960
        camera_lat = 35.182491

        lmaPlot.ax_plan.scatter(camera_lon, camera_lat, 100, 'goldenrod', '*', linewidths=.5, edgecolors='k', transform=ccrs.PlateCarree(), zorder=4)

        plot_points(lmaPlot, lonSet, latSet, altSet, timeSet, 'rainbow', 15, vmin, vmax, relative_colors, 'k', 0.25)

        savePath = path.join(outbasepath, str(i), 'lma.png')
        Path(path.dirname(savePath)).mkdir(parents=True, exist_ok=True)
        lmaPlot.fig.savefig(savePath)
        for el in np.arange(1, 30, 1.5):
            this_el_times = rax_times[f'{el:.1f}']
            # preceding_scan_time = [scan_time for scan_time in this_el_times if scan_time < time_start_dt][-1]
            # try:
            #     following_scan_time = [scan_time for scan_time in this_el_times if scan_time > time_start_dt][0]
            # except IndexError:
            #     continue
            # for field in radar_fields:
            #     raxPrecDataPath = path.join(basepath, 'rax', f'RAXPOL-{preceding_scan_time.strftime('%Y%m%d-%H%M%S')}-E{el:.1f}-{field["suffix"]}.nc')
            #     raxPrecData = xr.open_dataset(raxPrecDataPath)
            #     raxFollowDataPath = path.join(basepath, 'rax', f'RAXPOL-{following_scan_time.strftime('%Y%m%d-%H%M%S')}-E{el:.1f}-{field["suffix"]}.nc')
            #     raxFollowData = xr.open_dataset(raxFollowDataPath)


            #     badPracticeContainer = [raxPrecData, '', raxFollowData]

            #     fig, axs = plt.subplots(1, 3, subplot_kw={'projection' : ccrs.PlateCarree()}, figsize=(15, 5))

            #     for i in [0, 2]:
            #         data = badPracticeContainer[i]
            #         ax = axs[i]
            #         t_start = pd.Timestamp(np.array([data.Time]).astype('datetime64[s]')[0]).to_pydatetime()
            #         rng = np.matmul(data.Gate.data.reshape(-1, 1), data.GateWidth.data.reshape(1, -1))
            #         az = np.tile(data.Azimuth.data, (rng.shape[0], 1))
            #         eldat = np.tile(data.Elevation.data, (rng.shape[0], 1))
            #         rcs = RadarCoordinateSystem(data.Latitude, data.Longitude, rax_elevation_m)
            #         lon, lat, alt = rcs.toLonLatAlt(rng, az, eldat)
            #         lon = np.array(lon).reshape(az.shape)
            #         lat = np.array(lat).reshape(az.shape)
            #         alt = np.array(alt).reshape(az.shape)


            #         colorvar = np.ma.masked_array(data[field['varname']].data, data[field['varname']].data <= -99900.0).T

            #         handle = ax.pcolormesh(lon, lat, colorvar, vmin=field['vmin'], vmax=field['vmax'], cmap=field['cmap'], transform=ccrs.PlateCarree())
            #         fig.colorbar(handle, label=field['label'], aspect=90)
            #         ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='gray', linewidth=0.5)

            #         ax.scatter(lonSet, latSet, s=3, c='k', marker='.', transform=ccrs.PlateCarree(), alpha=0.1)

            #         ax.scatter(camera_lon, camera_lat, 100, 'goldenrod', '*', linewidths=.5, edgecolors='k', transform=ccrs.PlateCarree(), zorder=4)

            #         ax.set_title(f'RaXPol {t_start.strftime("%H:%M:%S")} UTC\nTarget elevation: {el:.1f}°\nActual elevation at az=0°: {data.isel(Azimuth=np.argmin(data.Azimuth.data)).Elevation.data:.1f}°')

            #         ax.set_xlim(lon_range)
            #         ax.set_ylim(lat_range)

            #     lightningAx = axs[1]
            #     deltaTs = (pd.to_datetime(timeSet) - pd.to_datetime(timeSet).reset_index(drop=True)[0]).dt.total_seconds().values
            #     handle = lightningAx.scatter(lonSet, latSet, s=3, c=deltaTs, cmap='rainbow', transform=ccrs.PlateCarree(), alpha=0.5)
            #     fig.colorbar(handle, label='Seconds since initiation', aspect=90)
            #     lightningAx.set_title(f'Lightning Flash\n{time_start_dt.strftime('%H:%M:%S.%f')} - {time_end_dt.strftime('%H:%M:%S.%f')} UTC')
            #     lightningAx.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='gray', linewidth=0.5)
            #     lightningAx.set_xlim(lon_range)
            #     lightningAx.set_ylim(lat_range)


            #     radarOutPath = path.join(path.dirname(savePath), f'{el:.1f}', f'{field["suffix"]}.png')
            #     Path(path.dirname(radarOutPath)).mkdir(parents=True, exist_ok=True)
            #     fig.tight_layout()
            #     fig.savefig(radarOutPath)

        pxpreceding_scan_time = [scan_time for scan_time in px_times if scan_time < time_start_dt][-1]
        pxfollowing_scan_time = [scan_time for scan_time in px_times if scan_time > time_start_dt][0]
        for field in radar_fields:
            pxPrecFileName = [file for file in sorted(listdir('px1k')) if file.startswith(pxpreceding_scan_time.strftime('PX-%Y%m%d-%H%M%S-A')) and file.endswith(f'-{field['suffix']}.nc')][0]
            pxPrecDataPath = path.join(basepath, 'px1k', pxPrecFileName)
            pxPrecData = xr.open_dataset(pxPrecDataPath)
            pxFollowFileName = [file for file in sorted(listdir('px1k')) if file.startswith(pxfollowing_scan_time.strftime('PX-%Y%m%d-%H%M%S-A')) and file.endswith(f'-{field['suffix']}.nc')][0]
            pxFollowDataPath = path.join(basepath, 'px1k', pxFollowFileName)
            pxFollowData = xr.open_dataset(pxFollowDataPath)

            pxContainer = [pxPrecData, pxFollowData]

            fig2 = plt.figure(figsize=(15, 5))
            ax1 = fig2.add_subplot(1, 3, 1)
            lightningAx = fig2.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
            ax3 = fig2.add_subplot(1, 3, 3)
            axs = [ax1, ax3]
            
            lon_rhi = np.array([])
            lat_rhi = np.array([])
            for i in [0, 1]:
                data = pxContainer[i]
                ax = axs[i]
                t_start = pd.Timestamp(np.array([data.Time]).astype('datetime64[s]')[0]).to_pydatetime()
                rng = np.matmul(data.Gate.data.reshape(-1, 1), data.GateWidth.data.reshape(1, -1))
                az = np.tile(data.Azimuth.data, (rng.shape[0], 1))
                eldat = np.tile(data.Elevation.data, (rng.shape[0], 1))

                rcs = RadarCoordinateSystem(data.Latitude, data.Longitude, px_elevation_m)
                tpcs = TangentPlaneCartesianSystem(data.Latitude, data.Longitude, px_elevation_m)
                ecef_X, ecef_Y, ecef_Z = rcs.toECEF(rng, az, eldat)
                tpcs_x, tpcs_y, tpcs_z = tpcs.fromECEF(ecef_X, ecef_Y, ecef_Z)

                tpcs_x = tpcs_x.reshape(rng.shape)
                tpcs_y = tpcs_y.reshape(rng.shape)
                tpcs_z = tpcs_z.reshape(rng.shape)

                ground_range = (tpcs_x**2 + tpcs_y**2)**0.5

                colorvar = np.ma.masked_array(data[field['varname']].data, data[field['varname']].data <= -99900.0).T

                handle = ax.pcolormesh(ground_range/1000, tpcs_z/1000, colorvar, vmin=field['vmin'], vmax=field['vmax'], cmap=field['cmap'])
                fig2.colorbar(handle, label=field['label'], aspect=90)

                ax.set_xlabel('Distance from radar (km)')
                ax.set_ylabel('Altitude (km)')

                ax.set_ylim(0, 13)

                ax.set_title(f'PX-1000 {t_start.strftime("%H:%M:%S")} UTC\n{data.Azimuth.data[0]:.1f}° RHI')
                scan_lon, scan_lat, _ = rcs.toLonLatAlt(rng, az, eldat)
                lon_rhi = np.append(lon_rhi, scan_lon).flatten()
                lat_rhi = np.append(lat_rhi, scan_lat).flatten()
            
            deltaTs = (pd.to_datetime(timeSet) - pd.to_datetime(timeSet).reset_index(drop=True)[0]).dt.total_seconds().values
            handle = lightningAx.scatter(lonSet, latSet, s=3, c=deltaTs, cmap='rainbow', transform=ccrs.PlateCarree(), alpha=0.5)
            fig2.colorbar(handle, label='Seconds since initiation', aspect=90)
            lightningAx.set_title(f'Lightning Flash\n{time_start_dt.strftime('%H:%M:%S.%f')} - {time_end_dt.strftime('%H:%M:%S.%f')} UTC')
            lightningAx.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='gray', linewidth=0.5)
            lightningAx.set_xlim(lon_range)
            lightningAx.set_ylim(lat_range)
            lightningAx.scatter(lon_rhi, lat_rhi, s=3, c='k', transform=ccrs.PlateCarree())

            radarOutPath = path.join(path.dirname(savePath), 'RHI', f'{field["suffix"]}.png')
            Path(path.dirname(radarOutPath)).mkdir(parents=True, exist_ok=True)
            fig2.tight_layout()
            fig2.savefig(radarOutPath)




