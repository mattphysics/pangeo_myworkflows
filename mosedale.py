from datetime import datetime
from dask.diagnostics.progress import ProgressBar
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import cartopy.crs as ccrs
import dask

import os, sys

import xskillscore as xs

import matplotlib.path as mpath
import matplotlib.pyplot as plt

sys.path.append('/network/group/aopp/oceans/LZ001_AENGENHEYSTER_OHUFLUX/stack/Oxford/OceanHeatUptake/PythonScripts')
import oceanHeatFunctions as ohf

import statsmodels.api as sm
from statsmodels.graphics import tsaplots

# ============
# Plotting
# ============

def plot_nao_corr_maps(ds,title=None,filename=None,lev_psl=np.arange(-0.75,0.75+0.15,0.15),lev_sst=np.arange(-0.4,0.4+0.1,0.1),psl_title='NAO-psl correlation',sst_title='NAO-SST correlation',figsize=(10,5)):
    fig = plt.figure(figsize=figsize)
    ax = [
        fig.add_subplot(121,projection=ccrs.LambertConformal(central_latitude=45,central_longitude=-40)),
        fig.add_subplot(122,projection=ccrs.LambertConformal(central_latitude=45,central_longitude=-55))
    ]

    xlim0 = [-100,20]
    # ylim0 = [0,90]
    ylim0 = [0,85]

    xlim1 = [-100,-10]
    ylim1 = [0,70]
    lower_space = 2 # this needs to be manually increased if the lower arched is cut off by changing lon and lat lims

    set_map_boundaries(xlim0,ylim0,lower_space,ax=ax[0])
    set_map_boundaries(xlim1,ylim1,lower_space,ax=ax[1])

    [axi.gridlines() for axi in ax]
    [axi.coastlines() for axi in ax]

    ds['r_NAO_psl'].sel(latitude=slice(90,0)).plot.contourf(levels=lev_psl,ax=ax[0],transform=ccrs.PlateCarree(),cbar_kwargs={'shrink':0.8,'orientation':'horizontal','pad':0,'format':'%.2f'})
    ds['r_NAO_sst'].sel(latitude=slice(90,0)).plot.contourf(levels=lev_sst,ax=ax[1],transform=ccrs.PlateCarree(),cbar_kwargs={'shrink':0.8,'orientation':'horizontal','pad':0,'format':'%.2f'})

    ax[0].set_title(psl_title)
    ax[1].set_title(sst_title)

    fig.tight_layout()
    fig.suptitle(title)

    if filename is not None:
        fig.savefig(filename)

    return fig, ax

from scipy.special import erfinv
# def correlation_convidence_interval(r,N,interval=0.95):
def correlation_convidence_interval(r,N,alpha=0.05):
    '''
    Compute confidence interval of a correlation value given
        r: correlation coefficient
        N: number of samples
        # interval: confidence interval, default: 0.95
        alpha: confidence interval, default: 0.05, corresponding to 95% confidence interval

    Uses Fischer transformation and inverse error function to get the critical Z-score
    '''
    Frho_mean = np.arctanh(r)
    Frho_se = (1 / (N - 3))**0.5
    # Z_crit = np.sqrt(2)*erfinv(interval)
    Z_crit = np.sqrt(2)*erfinv(1-alpha)
    conf_int = [np.tanh(Frho_mean - Z_crit * Frho_se),np.tanh(Frho_mean + Z_crit * Frho_se)]
    return conf_int


def auto_corr(x,dt,max=100,alpha=None,unit='days',**kwargs):
    '''
    Computes autocorrelation of variable x with time-spacing <dt> of unit <unit>
    Return <nlags> lags
    '''
    nlags=max//dt
    s = datoseries(x,step=dt)
    lags = s.index[:nlags+1]
    acf = sm.tsa.acf(s,alpha=alpha,nlags=nlags,**kwargs)
    if alpha is None:
        out = xr.DataArray(dims=['lag'],coords={'lag':lags},data=acf,name='acf_%s' % x.name)
    else:
        out = xr.Dataset(coords={'lag':lags})
        out['acf_%s' % x.name] = xr.DataArray(dims=['lag'],coords={'lag':lags},data=acf[0])
        out['acf_%s_lower' % x.name] = xr.DataArray(dims=['lag'],coords={'lag':lags},data=acf[1][:,0])
        out['acf_%s_upper' % x.name] = xr.DataArray(dims=['lag'],coords={'lag':lags},data=acf[1][:,1])
    out['lag'].attrs['units'] = unit
    out.attrs['description'] = 'Autocorrelation function for %s' % x.name
    out.attrs['alpha'] = alpha
    out.attrs['dt'] = dt
    return out



def lag_corr(x,y,dt,max=100,alpha=None,unit='days',**kwargs):
    '''
    Computes lagged cross-correlation between x[t] and y[t]: ccf(x,y)[k] = corr(x[t+k],y[t])

    Positive lags: x lags y: future values of x related to present values of y
    Negative lags: x leads y: past values of x related to present values of y

    x and y given with time-spacing of <dt> <unit> (e.g. 5 days for dt=5, unit=days)

    Returns values between += max lag

    alpha: uses Fischer transformation and inverse error function to get the critical Z-score
    '''
    forward  = sm.tsa.stattools.ccf(datoseries(x,step=dt),datoseries(y,step=dt),**kwargs) # x lags y
    backward = sm.tsa.stattools.ccf(datoseries(y,step=dt),datoseries(x,step=dt),**kwargs)[::-1] # x leads yl
    lags = np.array(np.arange(-(backward.size-1),0).tolist() + np.arange(forward.size).tolist()) * dt

    ccf = np.r_[backward[:-1],forward]
    if alpha is None:
        out = xr.DataArray(dims=['lag'],coords={'lag':lags},data=ccf,name='ccf_%s_%s' % (x.name,y.name)).sel(lag=slice(-max,max))
    else:
        N_lags = x.size - abs(lags) / dt
        lower, upper = correlation_convidence_interval(ccf,N_lags,alpha=alpha)
        # print(x.size)
        # print(lower)
        # print(N_lags)

        out = xr.Dataset(coords={'lag':lags})
        out['ccf_%s_%s' % (x.name,y.name)] = xr.DataArray(dims=['lag'],coords={'lag':lags},data=ccf)
        out['ccf_%s_%s_lower' % (x.name,y.name)] = xr.DataArray(dims=['lag'],coords={'lag':lags},data=lower)
        out['ccf_%s_%s_upper' % (x.name,y.name)] = xr.DataArray(dims=['lag'],coords={'lag':lags},data=upper)
        out['N_lags'] = xr.DataArray(dims=['lag'],coords={'lag':lags},data=N_lags)

        out = out.sel(lag=slice(-max,max))

    out['lag'].attrs['units'] = unit
    out.attrs['description'] = 'Positive lags: %s lags' % x.name
    return out

def plot_corr_lags(x,y,dt,max=100,alpha=None,unit='days',title=None,ylim_ccf=None,figsize=(8,3)):
    acf_x = auto_corr(x,dt=dt,max=max,alpha=alpha,unit=unit)
    acf_y = auto_corr(y,dt=dt,max=max,alpha=alpha,unit=unit)
    ccf_xy = lag_corr(x,y,dt=dt,max=max,alpha=alpha,unit=unit)

    fig, ax = plt.subplots(1,3,sharex=False,sharey=False,figsize=figsize)
    if alpha is None:
        acf_x.plot.line('.-',ax=ax[0],_labels=False)#, lw=3)
        acf_y.plot.line('.-',ax=ax[1],_labels=False)#, lw=3)
        ccf_xy.plot.line('.-',ax=ax[2],_labels=False)#, lw=3)
    else:
        acf_x['acf_%s' % x.name].plot.line('.-',ax=ax[0],_labels=False)#, lw=1)
        acf_y['acf_%s' % y.name].plot.line('.-',ax=ax[1],_labels=False)#, lw=2)
        ccf_xy['ccf_%s_%s' % (x.name,y.name)].plot.line('.-',ax=ax[2],_labels=False)#, lw=1)
        # Shading
        ax[0].fill_between(acf_x.lag,acf_x['acf_%s_lower' % x.name], acf_x['acf_%s_upper' % x.name],alpha=0.5)
        ax[1].fill_between(acf_y.lag,acf_y['acf_%s_lower' % y.name], acf_y['acf_%s_upper' % y.name],alpha=0.5)
        ax[2].fill_between(ccf_xy.lag,ccf_xy['ccf_%s_%s_lower' % (x.name,y.name)], ccf_xy['ccf_%s_%s_upper' % (x.name,y.name)],alpha=0.5)

    [axi.set_xlim(0,max) for axi in ax[:2]]
    ax[2].set_xlim(-max,max)

    ylims = [ax[0].get_ylim()[0],ax[1].get_ylim()[0]]
    [axi.set_ylim(bottom=min(ylims)) for axi in ax[:2]]
    ax[1].set_yticklabels([])
    ax[2].set_ylim(ylim_ccf)

    ax[0].set_title('%s autocorrelation' % x.name)
    ax[1].set_title('%s autocorrelation' % y.name)
    # ax[2].set_title('Lag-correlation %s-%s' % (x.name,y.name),loc='center')
    ax[2].set_title('%s leads' % x.name,loc='left')
    ax[2].set_title('%s leads' % y.name,loc='right')

    ax[0].set_ylabel('Correlation')
    [axi.set_xlabel('lag (%s)' % unit) for axi in ax]
    [axi.grid(True) for axi in ax]
    ax[2].axvline(c='k')
    fig.suptitle(title)
    fig.tight_layout(pad=0.5)

    return fig,ax

def plot_nao_corr_lags(ds,dt,max=100,alpha=None,title=None,labels=None,ylim_ccf=None,figsize=(8,3)):
    '''
    Plot the autocorrelation functions of NAO and Tripole indices, and their cross correlation

    dt gives the time-resolution of the input (standard: 5 days). THIS IS NOT CHECKED

    95% confidence intervals for autocorrelation
    '''
    fig, ax = plt.subplots(1,3,sharex=False,sharey=False,figsize=figsize)

    if isinstance(ds,list):
        for dsi in ds:
            # NAO autocorrelation
            acf_NAO = auto_corr(dsi['NAO'],dt=dt,max=max,alpha=alpha)
            if alpha is None:
                acf_NAO.plot.line('.-',ax=ax[0],_labels=False)
            else:
                acf_NAO['acf_NAO'].plot.line('.-',ax=ax[0],_labels=False)
                ax[0].fill_between(acf_NAO.lag,acf_NAO['acf_NAO_lower'], acf_NAO['acf_NAO_upper'],alpha=0.5)

            # Tripole autocorrelation
            acf_T = auto_corr(dsi['Tripole'],dt=dt,max=max,alpha=alpha)
            if alpha is None:
                acf_T.plot.line('.-',ax=ax[1],_labels=False)
            else:
                acf_T['acf_Tripole'].plot.line('.-',ax=ax[1],_labels=False)
                ax[1].fill_between(acf_T.lag,acf_T['acf_Tripole_lower'], acf_T['acf_Tripole_upper'],alpha=0.5)
            
            ccf_NAO_T = lag_corr(dsi['NAO'],dsi['Tripole'],dt=dt,max=max,alpha=alpha)
            if alpha is None:
                ccf_NAO_T.plot.line('.-',ax=ax[2],_labels=False)
            else:
                ccf_NAO_T['ccf_NAO_Tripole'].plot.line('.-',ax=ax[2],_labels=False)
                ax[2].fill_between(ccf_NAO_T.lag,ccf_NAO_T['ccf_NAO_Tripole_lower'], ccf_NAO_T['ccf_NAO_Tripole_upper'],alpha=0.5)
        if labels is not None:
            ax[0].legend(labels)
    else:
        # NAO autocorrelation
        acf_NAO = auto_corr(ds['NAO'],dt=dt,max=max,alpha=alpha)
        if alpha is None:
            acf_NAO.plot.line('.-',ax=ax[0],_labels=False)
        else:
            acf_NAO['acf_NAO'].plot.line('.-',ax=ax[0],_labels=False)
            ax[0].fill_between(acf_NAO.lag,acf_NAO['acf_NAO_lower'], acf_NAO['acf_NAO_upper'],alpha=0.5)

        # Tripole autocorrelation
        acf_T = auto_corr(ds['Tripole'],dt=dt,max=max,alpha=alpha)
        if alpha is None:
            acf_T.plot.line('.-',ax=ax[1],_labels=False)
        else:
            acf_T['acf_Tripole'].plot.line('.-',ax=ax[1],_labels=False)
            ax[1].fill_between(acf_T.lag,acf_T['acf_Tripole_lower'], acf_T['acf_Tripole_upper'],alpha=0.5)
        
        ccf_T_NAO = lag_corr(ds['Tripole'],ds['NAO'],dt=dt,max=max,alpha=alpha)
        if alpha is None:
            ccf_T_NAO.plot.line('.-',ax=ax[2],_labels=False)
        else:
            ccf_T_NAO['ccf_Tripole_NAO'].plot.line('.-',ax=ax[2],_labels=False)
            ax[2].fill_between(ccf_T_NAO.lag, ccf_T_NAO['ccf_Tripole_NAO_lower'], ccf_T_NAO['ccf_Tripole_NAO_upper'],alpha=0.5)

    ax[2].axvline(c='k')
    ax[0].set_title('NAO autocorrelation')
    ax[1].set_title('Tripole autocorrelation')
    # ax[2].set_title('Lag-correlation SST-NAO \npositive lags: NAO leads')
    ax[2].set_title('Tripole leads',loc='right')
    ax[2].set_title('NAO leads',loc='left')


    [axi.set_xlim(0,max) for axi in ax[:2]]
    ax[2].set_xlim(-max,max)

    ylims = [ax[0].get_ylim()[0],ax[1].get_ylim()[0]]
    [axi.set_ylim(bottom=min(ylims)) for axi in ax[:2]]
    ax[1].set_yticklabels([])
    ax[2].set_ylim(ylim_ccf)

    [axi.grid(True) for axi in ax]

    ax[0].set_ylabel('Correlation')
    [axi.set_xlabel('lag (days)') for axi in ax]
    fig.suptitle(title)
    fig.tight_layout(pad=0.5)

    return fig,ax

# ============
# ============
def set_map_boundaries(xlim,ylim,lower_space,ax):
    '''
    Set map to remove whitespace for conic projections
    '''
    rect = mpath.Path([[xlim[0], ylim[0]],
                    [xlim[1], ylim[0]],
                    [xlim[1], ylim[1]],
                    [xlim[0], ylim[1]],
                    [xlim[0], ylim[0]],
                    ]).interpolated(20)

    proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
    rect_in_target = proj_to_data.transform_path(rect)

    ax.set_boundary(rect_in_target)
    ax.set_extent([xlim[0], xlim[1], ylim[0] - lower_space, ylim[1]])

def datoseries(da,name=None,step=1):
    '''Convert DataArray to pandas Series with Range Index with <step>'''
    if name is None and da.name is None:
        print('Need a name!')
        sys.exit()
    if name is not None:
        s = da.rename(name).to_dataframe().reset_index()[name]
    else:
        s = da.rename(da.name).to_dataframe().reset_index()[da.name]
    s.index = pd.RangeIndex(start=0,stop=s.size*step,step=step)
    return s

def preprocess_era(ds):
    '''Pre-process ERA5 ncfiles'''
    rename_dict = {}
    if 'lon' in ds.dims:
        rename_dict['lon'] = 'longitude'
    if 'lat' in ds.dims:
        rename_dict['lat'] = 'latitude'
    ds = ds.rename(rename_dict)
    # Some files have wrong latitude coordinate!
    if ds['latitude'].values[0] == -88.75:
        # ds = ds.interp(latitude=np.arange(90,-90-1,-1.))
        ds = ds.interp(latitude=np.arange(-90,90+2.5,2.5))
    
    return ds
def is_winter(month):
    '''helper function to select winter half of year (October-March)via da.sel(time=is_winter(da['time.month'])'''
    return (month >= 10) | (month <= 3)
def is_summer(month):
    '''helper function to select summer half of year (April-September) via da.sel(time=is_summer(da['time.month'])'''
    return (month >= 4) & (month <= 9)

def get_winter(da,time='time'):
    '''Restrict a DataArray to winter months (October-March) only'''
    return da.sel({time:is_winter(da['%s.month' % time])})
def get_summer(da,time='time'):
    '''Restrict a DataArray to summer months (April-September) only'''
    return da.sel({time:is_summer(da['%s.month' % time])})

def area_da(da):
    '''Get grid box area from DataArray'''
    other_dims = [d for d in da.dims if not 'latitude' in d and not 'longitude' in d]
    area = ohf.ocean_area(da.isel({d:0 for d in other_dims}),outin=True)
    area = area.drop([c for c in area.coords if not c in area.dims])
    area.attrs = {}
    return area

def weights_da(da):
    '''
    Convenience function
    Get area weights from DataArray
    Return grid box area normalized by sum over all grid boxes: area / area.sum()
    '''
    area = area_da(da)
    return area / area.sum()

def area_weighted_mean(da,area=None):
    '''Compute area-weighted mean of a DataArray (should also work for DataSets)'''
    if area is None:
        area = area_da(da)
    weights = area / area.sum()
    return (da * weights).sum(weights.dims)

def standardize(da,dims=None):
    '''
    Standardize a field by subtracting the mean and dividing by standard deviation
    If no dimensions are provided, compute mean and std.dev. over all dimensions
    '''
    if dims is None: # over all dims
        print('Standardizing over all dimensions')
        return (da - da.mean()) / da.std()
    else: # over provided dims
        print('Standardizing over these dimensions: %s' % dims)
        return (da - da.mean(dims)) / da.std(dims)

def nao_from_std(ds,varname='psl',latname='latitude',lonname='longitude',timename='timed'):
    '''
    Compute NAO index according to Mosedale et al 2005
    From "ERA5-like" grid, i.e. -90N - 90N, starting at the South Pole

    Use 5-daymeans of psl

    - take in normalized anomalies
    - difference between area-weighted averaged psl boxes (south minus north)
    '''

    box_south = ds[varname].sel({latname:slice(55,22),lonname:slice(-90,60)})
    box_north = ds[varname].sel({latname:slice(90,55),lonname:slice(-90,60)})

    box_south_mean = area_weighted_mean(box_south)
    box_north_mean = area_weighted_mean(box_north)

    nao_mose = box_south_mean - box_north_mean
    return nao_mose.rename('NAO')


def anomalies(ds,type='time',timename='time',relname='realization'):
    '''
    Compute anomalies from seasonal cycle
    1. type "time": assumes long multi-year timeseries, single realization: Climatology by averaging over dayofyear
    2. type "ensemble": assumes short single-year timeseries, multiple realizations: Climatology by ensemble mean
    '''
    if type == 'time':
        return ds.groupby('%s.dayofyear' % timename) - ds.groupby('%s.dayofyear' % timename).mean()
    elif type ==  'ensemble':
        return ds - ds.mean(relname)


def standardize_anom_season(ds,season=None,type='time',timename='time',relname='realization'):
    '''
    Compute standardized anomalies, selecting a season only if desired
    Order of operations:
    1. Compute anomalies of type "time" or type "ensemble"
    2. Standardize over time
    3. Select values from a given season
    '''

    # Winter anomalies: anomalies -> standardize -> restrict to winter
    ds_anom = anomalies(ds,type=type,timename=timename,relname=relname)
    ds_std = standardize(ds_anom,dims=timename)
    if season == 'winter':
        print('Get winter values only')
        return get_winter(ds_std,time=timename)
    elif season == 'summer':
        print('Get summer values only')
        return get_summer(ds_std,time=timename)
    elif season is None:
        print('No season selected')
        return ds_std
    else:
        print('Season %s is undefined.')
        return None

def mose_era_compute(res,time,season,save=True,return_ds=False):
    '''
    Compute Mosedale statistics from ERA5
    Inputs:
        -res: resolution: 2.5, 1, 0.5
        -time: time resolution: daily, 5daily
        -season: winter, summer
    '''
    # Load ERA5 data
    print('Loading data')
    ds = era5_load(res=res,time=time)
    # Standardized winter anomalies
    print('Standardizing data')
    ds = standardize_anom_season(ds,season=season,type='time',timename='time')
    # Compute NAO
    print('Computing NAO')
    nao = nao_from_std(ds,timename='time').load()
    # Compute patterns
    print('Computing pattern correlations')
    pattern_nao = xs.pearson_r(nao,ds['psl'].fillna(0.).load(),dim=['time']).rename('r_NAO_psl')
    pattern_sst = xs.pearson_r(nao,ds['sst'].fillna(0.).load(),dim=['time']).rename('r_NAO_sst')
    # Compute tripole index
    print('Computing pattern tripole')
    tripole = (ds['sst'] * pattern_sst.sel(longitude=slice(-95,-10),latitude=slice(70,0))).sum(['latitude','longitude']).rename('Tripole')
    # Merge & compute
    ds_out = xr.merge([
        nao,
        tripole,
        pattern_nao,
        pattern_sst
    ])
    with ProgressBar():
        ds_out.load()

    ds_out.attrs['description'] = 'NAO and SST Tripole indices computed according to Mosedale et al., 2006'
    ds_out.attrs['resolution'] = res
    ds_out.attrs['time'] = time
    ds_out.attrs['season'] = season
    ds_out.attrs['date'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

    # Save/return
    if save is True:
        if season is None:
            outname = '/network/group/aopp/oceans/LZ001_AENGENHEYSTER_OHUFLUX/Slab/analysis/mose_era5_%s_%s.nc' % (res,time)
        else:
            outname = '/network/group/aopp/oceans/LZ001_AENGENHEYSTER_OHUFLUX/Slab/analysis/mose_era5_%s_%s_%s.nc' % (res,time,season)
        print('Saving ERA5 statistics to %s' % outname)
        ds_out.to_netcdf(outname)
    if return_ds is True:
        print('Returning ')
        return ds, ds_out

def mose_slab_compute(exp,season,save=True,return_ds=False):
    '''
    Compute Mosedale statistics from ERA5
    Inputs:
        -res: resolution: 2.5, 1, 0.5
        -time: time resolution: daily, 5daily
        -season: winter, summer
    '''
    # Load ERA5 data
    print('Loading data')
    ds = mose_slab_control_preprocess(exp)
    # Standardized winter anomalies
    print('Standardizing data')
    ds = standardize_anom_season(ds,season=season,type='time',timename='time5d')
    # Compute NAO
    print('Computing NAO')
    nao = nao_from_std(ds,timename='time').load()
    # Compute patterns
    print('Computing pattern correlations')
    pattern_nao = xs.pearson_r(nao,ds['psl'].fillna(0.).load(),dim=['time5d']).rename('r_NAO_psl')
    pattern_sst = xs.pearson_r(nao,ds['sst'].fillna(0.).load(),dim=['time5d']).rename('r_NAO_sst')
    # Compute tripole index
    print('Computing pattern tripole')
    tripole = (ds['sst'] * pattern_sst.sel(longitude=slice(-95,-10),latitude=slice(70,0))).sum(['latitude','longitude']).rename('Tripole')
    # Merge & compute
    ds_out = xr.merge([
        nao,
        tripole,
        pattern_nao,
        pattern_sst
    ])
    with ProgressBar():
        ds_out.load()

    ds_out.attrs['description'] = 'NAO and SST Tripole indices computed according to Mosedale et al., 2006'
    ds_out.attrs['exp'] = exp
    ds_out.attrs['season'] = season
    ds_out.attrs['date'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

    # Save/return
    if save is True:
        if season is None:
            outname = '/network/group/aopp/oceans/LZ001_AENGENHEYSTER_OHUFLUX/Slab/analysis/mose_%s.nc' % (exp)
        else:
            outname = '/network/group/aopp/oceans/LZ001_AENGENHEYSTER_OHUFLUX/Slab/analysis/mose_%s_%s.nc' % (exp,season)
        print('Saving Slab ARC run statistics to %s' % outname)
        ds_out.to_netcdf(outname)
    if return_ds is True:
        print('Returning ')
        return ds, ds_out

def mose_era_load(res,time,season):
    '''
    Load computed & saved Mosedale statistics for ERA5
    Inputs:
        -res: resolution: 2.5, 1, 0.5
        -time: time resolution: daily, 5daily
        -season: winter, summer
    '''
    if season is None:
        outname = 'network/group/aopp/oceans/LZ001_AENGENHEYSTER_OHUFLUX/Slab/analysis/mose_era5_%s_%s.nc' % (res,time)
    else:
        outname = 'network/group/aopp/oceans/LZ001_AENGENHEYSTER_OHUFLUX/Slab/analysis/mose_era5_%s_%s_%s.nc' % (res,time,season)
    print('Loading ERA5 statistics from %s' % outname)
    ds_out = xr.open_dataset(outname)
    return ds_out


def mose_slab_control_preprocess(exp):
    '''
    1. Load .nc files, resample psl to 5-daily
    '''
    sst = xr.open_dataset('/network/group/aopp/oceans/LZ001_AENGENHEYSTER_OHUFLUX/Slab/results/%s/%sa.pc.tslab.1982-2012.nc' % (exp,exp))['temp'].rename('sst').sel(t=slice('1983',None))
    sst = sst.rename({'t':'time5d'}).squeeze(drop=True)
    psl = xr.open_dataset('/network/group/aopp/oceans/LZ001_AENGENHEYSTER_OHUFLUX/Slab/results/%s/%sa.pa.psl.1982-2012.nc' % (exp,exp))['p'].rename('psl').sel(t=slice('1983',None))
    psl = psl.resample({'t':'5D'},loffset=pd.Timedelta('2.5D')).mean().rename({'t':'time5d'}).squeeze(drop=True)

    ds_out = xr.merge([
        ohf.lon_360_to_180(psl),
        ohf.lon_360_to_180(sst).interp(time5d=psl['time5d'])
    ])

    return ds_out





def mose_cpdn_preprocess(ds):
    '''
    Preprocess CPDN ensemble data to compute Mosedale statistics
    1. Bin sea level pressure from daily to 5daily
    2. Interpolate SST (slab temperature) from time5d_bd to time5d
    '''

    # 1. Bin sea level pressure from daily to 5daily
    # psl = ds['psl'].groupby_bins('timed', ds['time5d_bd'].values).mean()
    psl = ds['psl'].resample({'timed':'5D'},loffset=pd.Timedelta('2.5D')).mean().rename({'timed':'time5d'})
    psl = psl.isel(time5d=slice(1,None))

    # 2. Interpolate SST (slab temperature) from time5d_bd to time5d
    sst = ds['tslab'].interp(time5d_bd=ds['time5d']).rename('sst')
    sst = sst.isel(time5d=slice(1,None)) # drop first timestep - no data for SST
    
    ds_out = xr.merge([
        psl,
        sst
    ])

    return ds_out

def mose_docile_preprocess(ds):
    '''
    Preprocess DOCILE ensemble data to compute Mosedale statistics
    1. Take NH 6-hourly sea level pressure and resample to 5D with 2.5D offset so it aligns with time5d
    2. Load OSTIA data (at time5d), for the time-period covered by psl
    '''
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        # 1. Take NH 6-hourly sea level pressure and resample to 5D with 2.5D offset so it aligns with time5d
        psl = ds['psl_NH'].squeeze().resample({'time6h_2':'5D'},loffset=pd.Timedelta('2.5D')).mean().rename({'time6h_2':'time5d'}).rename({'latitude_NH':'latitude'}).rename('psl')

        # 2. Load OSTIA data (at time5d), for the time-period covered by psl
        ostia_sst = ohf.lon_360_to_180(xr.open_dataarray('/network/group/aopp/oceans/LZ001_AENGENHEYSTER_OHUFLUX/Slab/ancils/OSTIA/OSTIA_sst_N144_0211-1701.nc').squeeze()).sel(t=slice('2003','2016')).rename({'t':'time5d'})
        # standardized anomalies (relative to the whole record)
        # ostia_sst_anom = anomalies(ostia_sst,type='time',timename='time5d')
        # ostia_sst_std = standardize(ostia_sst_anom,dims='time5d')
        ostia_sst_std = standardize_anom_season(ostia_sst,type='time',timename='time5d',season=None).rename('sst_std')
        ostia_sst_std_year = ostia_sst_std.sel(time5d=slice(*psl['time5d'].isel(time5d=[0,-1])))

        ostia_sst_year = ostia_sst.sel(time5d=slice(*psl['time5d'].isel(time5d=[0,-1]))).rename('sst')

        ds_out = xr.merge([
            psl,
            ostia_sst_year,
            ostia_sst_std_year
        ]).sel(latitude=slice(90,-90,-1))

        ds_out = get_winter(ds_out.broadcast_like(ds_out['psl']),time='time5d')
    return ds_out

def mose_cpdn_compute(ds,season,save=True,return_ds=False):
    '''
    Compute Mosedale statistics from CPDN ensemble
    Inputs:
        -res: resolution: 2.5, 1, 0.5
        -time: time resolution: daily, 5daily
        -season: winter, summer
    '''

    # Load batch data
    batch = ds.attrs['batch']
    print('Loading data')
    ds = mose_cpdn_preprocess(ds)
    # Standardized winter anomalies
    print('Standardizing data')
    # ds = standardize_anom_season(ds,season=season,type='ensemble',timename='time5d') # standardize only over time
    ds = get_winter(standardize(anomalies(ds,type='ensemble',timename='time5d'),dims=['time5d','realization']),time='time5d') # standardize over time and realization
    # Compute NAO
    print('Computing NAO')
    with ProgressBar():
        nao = nao_from_std(ds,timename='time5d').load()
    # Compute patterns
    print('Computing pattern correlations')
    pattern_nao = xs.pearson_r(nao,ds['psl'].fillna(0.).load(),dim=['time5d','realization']).rename('r_NAO_psl')
    pattern_sst = xs.pearson_r(nao,ds['sst'].fillna(0.).load(),dim=['time5d','realization']).rename('r_NAO_sst')
    # Compute tripole index
    print('Computing pattern tripole')
    tripole = (ds['sst'] * pattern_sst.sel(longitude=slice(-95,-10),latitude=slice(70,0))).sum(['latitude','longitude']).rename('Tripole')
    # Merge & compute
    ds_out = xr.merge([
        nao,
        tripole,
        pattern_nao,
        pattern_sst
    ])
    with ProgressBar():
        ds_out.load()

    ds_out.attrs['description'] = 'NAO and SST Tripole indices computed according to Mosedale et al., 2006'
    ds_out.attrs['season'] = season
    ds_out.attrs['date'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

    # Save/return
    if save is True:
        if season is None:
            outname = '/network/group/aopp/oceans/LZ001_AENGENHEYSTER_OHUFLUX/Slab/analysis/mose_batch_%s.nc' % batch
        else:
            outname = '/network/group/aopp/oceans/LZ001_AENGENHEYSTER_OHUFLUX/Slab/analysis/mose_batch_%s_%s.nc' % (batch,season)
        print('Saving CPDN statistics to %s' % outname)
        ds_out.to_netcdf(outname)
    if return_ds is True:
        print('Returning ')
        return ds, ds_out


def mose_docile_compute(ds,season,save=True,return_ds=False):
    '''
    Compute Mosedale statistics from DOCILE ensemble
    Inputs:
        -res: resolution: 2.5, 1, 0.5
        -time: time resolution: daily, 5daily
        -season: winter, summer
    '''

    # Load batch data
    batch = ds.attrs['batch']
    ds = mose_docile_preprocess(ds)
    # Standardized winter anomalies: 1) Run 'standardize_anom_season' on ensemble sea level pressure; 2) take OSTIA standardized anomalies for that year
    ds = xr.merge([
        # standardize_anom_season(ds['psl'],season=season,type='ensemble',timename='time5d'),
        get_winter(standardize(anomalies(ds['psl'],type='ensemble',timename='time5d'),dims=['time5d','realization']),time='time5d'),
        ds['sst_std'].rename('sst')
        ])
    # ds = standardize_anom_season(ds,season=season,type='ensemble',timename='time5d')
    # Compute NAO
    with ProgressBar():
        nao = nao_from_std(ds,timename='time5d').load()
    # Compute patterns
    pattern_nao = xs.pearson_r(nao,ds['psl'].fillna(0.).load(),dim=['time5d','realization']).rename('r_NAO_psl')
    pattern_sst = xs.pearson_r(nao,ds['sst'].fillna(0.).load(),dim=['time5d','realization']).rename('r_NAO_sst')
    # Compute tripole index
    tripole = (ds['sst'] * pattern_sst.sel(longitude=slice(-95,-10),latitude=slice(70,0))).sum(['latitude','longitude']).rename('Tripole')
    # Merge & compute
    ds_out = xr.merge([
        nao,
        tripole,
        pattern_nao,
        pattern_sst
    ])
    with ProgressBar():
        ds_out.load()

    ds_out.attrs['description'] = 'NAO and SST Tripole indices computed according to Mosedale et al., 2006'
    ds_out.attrs['season'] = season
    ds_out.attrs['date'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

    # Save/return
    if save is True:
        year = ds_out['time5d.year'][0].values
        if season is None:
            outname = '/network/group/aopp/oceans/LZ001_AENGENHEYSTER_OHUFLUX/Slab/analysis/mose_batch_%s_%i.nc' % (batch,year)
        else:
            outname = '/network/group/aopp/oceans/LZ001_AENGENHEYSTER_OHUFLUX/Slab/analysis/mose_batch_%s_%i_%s.nc' % (batch,year,season)
        print('Saving DOCILE statistics to %s' % outname)
        ds_out.to_netcdf(outname)
    if return_ds is True:
        print('Returning ')
        return ds, ds_out




# ============

def da_timeseries_to_sns(da,time):
    '''
    Convert a DataArray timeseries to a long-form pandas Dataframe and add a <dt_days> coordinate that is the difference in days since the first timestep (i.e. df['dt_days'][0] == 0)
    Does allow more mulitple spatial coordinates - but be careful, DataFrame could become very large very quickly
    '''
    df = da.to_dataframe().reset_index()
    df['dt_days'] = (df[time] - df[time][0]).astype('m8[D]')
    return df

def pairwise_diff(da1,da2):
    '''Differences between all possible pairs in ensembles'''
    changes = []
    n = 1
    for mem1 in da1.realization.values:
        for mem2 in da2.realization.values:
            mem_out = '%.6d' % n
            change = da2.sel(realization=mem2) - da1.sel(realization=mem1) 
            change.coords['realization'] = mem_out
            changes.append(change)
    changes = xr.concat(changes,dim='realization')
    return changes

def plot_map1(da,title='',projection=ccrs.PlateCarree()):
    '''Simple map plot of one lat-lon DataArray'''
    plt.figure()
    ff = da.plot(subplot_kws={'projection':projection},transform=ccrs.PlateCarree())
    ff.axes.coastlines()
    ff.axes.set_title(title)
    return ff


