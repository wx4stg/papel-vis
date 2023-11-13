from os import listdir, path
import pyart
from matplotlib import pyplot as plt
from matplotlib import colors as pltcolors
import matplotlib as mpl
from cartopy import crs as ccrs
from cartopy import feature as cfeat
from metpy.plots import USCOUNTIES
import xarray as xr
import numpy as np
from pyxlma.coords import RadarCoordinateSystem, TangentPlaneCartesianSystem
import pandas as pd
from scipy import stats
import multiprocessing as mp

import warnings
warnings.filterwarnings("ignore")


files = sorted(listdir('px1k'))
fileds = ['px1k/'+f for f in files if f.endswith('-D.nc')]
fileps = ['px1k/'+f for f in files if f.endswith('-P.nc')]
filers = ['px1k/'+f for f in files if f.endswith('-R.nc')]
filevs = ['px1k/'+f for f in files if f.endswith('-V.nc')]
filews = ['px1k/'+f for f in files if f.endswith('-W.nc')]
filezs = ['px1k/'+f for f in files if f.endswith('-Z.nc')]


rax_elevation_m = 380
px_elevation_m = 318

totalf = len(fileds)

def makeImage(k):
    if '-E' not in fileds[k] and '-A' not in fileds[k]:
        print('Non-ppi file '+fileds[k])
        return
    if '-E90.0' in fileds[k]:
        print('Vertically pointing file '+fileds[k])
        return
    print(f'Processing {len(listdir('pxout/'))}/{totalf}, {len(listdir('pxout/'))/totalf*100:.2f}%')
    d = xr.open_dataset(fileds[k])
    time = pd.Timestamp(np.array([d.Time]).astype('datetime64[s]')[0]).to_pydatetime()
    saveFilePath = time.strftime('pxout/%Y%m%d_%H%M%S.png')
    if path.exists(saveFilePath):
        return
    p = xr.open_dataset(fileps[k])
    p.PhiDP.data = np.ma.masked_array(p.PhiDP.data, p.PhiDP.data <= -99900.0)
    p.PhiDP.data = np.rad2deg(p.PhiDP.data)
    r = xr.open_dataset(filers[k])
    v = xr.open_dataset(filevs[k])
    w = xr.open_dataset(filews[k])
    z = xr.open_dataset(filezs[k])


    all_fields = [{
        'varname' : 'Intensity',
        'data' : z,
        'label' : r'$Z$ dBZ',
        'cmap' : 'pyart_ChaseSpectral',
        'vmin' : -10,
        'vmax' : 80    
    }, {
        'varname' : 'Radial_Velocity',
        'data' : v,
        'label' : r'$V_{r}$ m/s',
        'cmap' : 'pyart_balance',
        'vmin' : -v.attrs['Nyquist_Vel-value'],
        'vmax' : v.attrs['Nyquist_Vel-value']
    }, {
        'varname' : 'Differential_Reflectivity',
        'data' : d,
        'label' : r'$Z_{DR}$ dB',
        'cmap' : 'pyart_turbone_zdr',
        'vmin' : -2,
        'vmax' : 8
    }, {
        'varname' : 'RhoHV',
        'data' : r,
        'label' : r'$\rho_{HV}$',
        'cmap' : 'pyart_plasmidis',
        'vmin' : 0,
        'vmax' : 1
    }, {
        'varname' : 'Width',
        'data' : w,
        'label' : 'Spectrum Width m/s',
        'cmap' : 'cubehelix_r',
        'vmin' : 0,
        'vmax' : 12.5
    }, {
        'varname' : 'PhiDP',
        'data' : p,
        'label' : r'$\phi_{DP}$ deg',
        'cmap' : 'pyart_Wild25',
        'vmin' : -180,
        'vmax' : 180
    }]

    px = 1/plt.rcParams['figure.dpi']
    if '-E' in fileds[k]:
        fig, axs = plt.subplots(3, 2, subplot_kw={'projection': ccrs.LambertConformal()})
        idx = 0
        for i in range(3):
            for j in range(2):
                ax = axs[i, j]
                thisfield = all_fields[idx]
                data = thisfield['data']
                
                rng = np.matmul(data.Gate.data.reshape(-1, 1), data.GateWidth.data.reshape(1, -1))
                az = np.tile(data.Azimuth.data, (rng.shape[0], 1))
                el = np.tile(data.Elevation.data, (rng.shape[0], 1))
                rcs = RadarCoordinateSystem(data.Latitude, data.Longitude, 0)
                lon, lat, alt = rcs.toLonLatAlt(rng, az, el)
                lon = np.array(lon).reshape(az.shape)
                lat = np.array(lat).reshape(az.shape)
                alt = np.array(alt).reshape(az.shape)


                colorvar = np.ma.masked_array(data[thisfield['varname']].data, data[thisfield['varname']].data <= -99900.0).T

                handle = ax.pcolormesh(lon, lat, colorvar, vmin=thisfield['vmin'], vmax=thisfield['vmax'], cmap=thisfield['cmap'], transform=ccrs.PlateCarree())
                ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='gray', linewidth=0.5)
                fig.colorbar(handle, label=thisfield['label'])
                idx = idx+1
        fig.suptitle(f'RaXPol PPI\nAvg. Elevation={np.mean(el):.2f}\n'+time.strftime('%d %b %Y %H:%M:%S UTC'))
        fig.set_size_inches(1024*px, 1050*px)
        fig.tight_layout()
        fig.savefig(saveFilePath)
    elif '-A' in fileds[k]:
        fig, axs = plt.subplots(3, 2)
        idx = 0
        for i in range(3):
            for j in range(2):
                ax = axs[i, j]
                thisfield = all_fields[idx]
                data = thisfield['data']

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

                colorvar = np.ma.masked_array(data[thisfield['varname']].data, data[thisfield['varname']].data <= -99900.0).T

                handle = ax.pcolormesh(ground_range, tpcs_z, colorvar, vmin=thisfield['vmin'], vmax=thisfield['vmax'], cmap=thisfield['cmap'])
                fig.colorbar(handle, label=thisfield['label'])
                idx = idx+1
        fig.suptitle(f'PX-1000 RHI\nAz={np.mean(az):.2f}\n'+time.strftime('%d %b %Y %H:%M:%S UTC'))
        fig.set_size_inches(1024*px, 1050*px)
        fig.tight_layout()
        fig.savefig(saveFilePath)



if __name__ == '__main__':
    listOfIdcs = sorted(list(range(totalf)))
    with mp.Pool(6) as p:
        p.map(makeImage, listOfIdcs)