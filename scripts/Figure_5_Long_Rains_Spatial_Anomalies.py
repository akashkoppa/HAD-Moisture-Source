#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Akash Koppa

"""
#%% load required libraries
import geopandas as gp
import pandas as pd
import pickle as pi
import matplotlib as mp
import cartopy as ca
import os as os
import numpy as np
import xarray as xr
import palettable as pl
import cmcrameri as cm
import xesmf as xe

#%% read in the hoa masks
shp_hoa = gp.read_file("/Stor1/horn_of_africa/input/study_region_masks/mask_hoa/mask_hoa.shp")

#%% read in the mask
maskfi = "/Stor1/horn_of_africa/input/study_region_masks/mask_hoa.nc"
maskda = xr.open_dataset(maskfi)
maskda = maskda["mask"]

#%% read in the flexpart data 
flxfil = "/Stor1/horn_of_africa/input/hamster/drylands/daily"
ds = xr.open_mfdataset(os.path.join(flxfil, "*.nc"), chunks = {"lat": 50, "lon": 50, "time" :-1})

#%% read in the wind data 
ds_uv = xr.open_dataset("/Stor1/horn_of_africa/input/era5/input/u_v_monthly.nc")
ds_uv = ds_uv.sel(expver = 1)

#%% regrid the wind data to the flexpart grid
regridder = xe.Regridder(ds_uv, maskda, "nearest_s2d")
ds_uv = regridder(ds_uv)

#%% calculate precipitation totals 
# long rain
pre_month = ds["E2P_EPs"].resample(time = "1M").sum()
pre_lr = pre_month.sel(time = pre_month.time.dt.month.isin([3,4,5]))
pre_lr.compute()
pre_lr_annual = pre_lr.groupby("time.year").sum()
pre_lr_annual_anm = pre_lr_annual - pre_lr_annual.mean(dim = "year")
# wet years
pre_lr_annual_wet = pre_lr_annual.sel(year = [1981, 1985, 1987, 1988, 1990]).mean(dim = "year")
# 95% of the rainfall
advtmp = np.array(pre_lr_annual_wet)
advtmp_vec = advtmp.flatten()
advtmp_vec = np.sort(advtmp_vec)[::-1]
advtmp_per = (advtmp_vec/np.sum(advtmp_vec))*100
advtmp_per = np.cumsum(advtmp_per)
tmp_q95 = advtmp_vec[np.where(advtmp_per <= 95)]
pre_lr_annual_wet_q95_mask = xr.where(pre_lr_annual_wet > tmp_q95[tmp_q95.size - 1], 1.0, np.nan)
pre_lr_annual_wet_q95 = xr.where(pre_lr_annual_wet_q95_mask == 1, pre_lr_annual_wet, np.nan)
pre_lr_annual_wet_q95_anm = pre_lr_annual_wet_q95 - pre_lr_annual.mean(dim = "year")


# dry years
pre_lr_annual_dry = pre_lr_annual.sel(year = [1999, 2004, 2008, 2009, 2011]).mean(dim = "year")
# 95% of the rainfall
advtmp = np.array(pre_lr_annual_dry)
advtmp_vec = advtmp.flatten()
advtmp_vec = np.sort(advtmp_vec)[::-1]
advtmp_per = (advtmp_vec/np.sum(advtmp_vec))*100
advtmp_per = np.cumsum(advtmp_per)
tmp_q95 = advtmp_vec[np.where(advtmp_per <= 95)]
pre_lr_annual_dry_q95_mask = xr.where(pre_lr_annual_dry > tmp_q95[tmp_q95.size - 1], 1.0, np.nan)
pre_lr_annual_dry_q95 = xr.where(pre_lr_annual_dry_q95_mask == 1, pre_lr_annual_dry, np.nan)
pre_lr_annual_dry_q95_anm = pre_lr_annual_dry_q95 - pre_lr_annual.mean(dim = "year")

#%% process the wind data
# wet 
ds_uv_wet = ds_uv.sel(time = ds_uv.time.dt.month.isin([3,4,5]))
ds_uv_wet = ds_uv_wet.sel(time = ds_uv_wet.time.dt.year.isin([1981, 1985, 1987, 1988, 1990]))
ds_uv_wet.compute()
ds_uv_wet = ds_uv_wet.groupby("time.year").mean()
ds_uv_wet = ds_uv_wet.mean(dim = ["year"])
# resample to have less number of arrows
lat_req = np.array(ds_uv_wet.lat)
lat_req = lat_req[lat_req >= -35]
lat_req = lat_req[lat_req <= 35]
ds_uv_wet = ds_uv_wet.sel(lat = lat_req)
ds_uv_wet = ds_uv_wet.isel(lon = np.arange(0, 360, 3), lat = np.arange(0, 71, 3))
ds_uv_wet = ds_uv_wet.drop("expver")

# dry
ds_uv_dry = ds_uv.sel(time = ds_uv.time.dt.month.isin([3,4,5]))
ds_uv_dry = ds_uv_dry.sel(time = ds_uv_dry.time.dt.year.isin([1999, 2004, 2008, 2009, 2011]))
ds_uv_dry.compute()
ds_uv_dry = ds_uv_dry.groupby("time.year").mean()
ds_uv_dry = ds_uv_dry.mean(dim = ["year"])
# resample to have less number of arrows
lat_req = np.array(ds_uv_dry.lat)
lat_req = lat_req[lat_req >= -35]
lat_req = lat_req[lat_req <= 35]
ds_uv_dry = ds_uv_dry.sel(lat = lat_req)
ds_uv_dry = ds_uv_dry.isel(lon = np.arange(0, 360, 3), lat = np.arange(0, 71, 3))
ds_uv_dry = ds_uv_dry.drop("expver")

#%% plot the four figures for long rains
# long rains (WET)
figure = mp.pyplot.figure(figsize = (6,6))
figaxi = figure.add_subplot(1, 1, 1, 
                            projection = ca.crs.Geostationary(central_longitude = 48))
figaxi.set_global()
figaxi.add_feature(ca.feature.LAND, edgecolor='black', alpha = 1.0, zorder =1)
figaxi.add_feature(ca.feature.OCEAN, zorder = 0)
figaxi.coastlines(zorder = 2)
figaxi.add_feature(ca.feature.BORDERS, zorder = 1, linestyle = "--", color = "black",
                   linewidth = 0.5)
figaxi.gridlines(crs=ca.crs.PlateCarree(), color = "black",
                  zorder = 1, linestyle = "--")
shp_hoa.plot(ax = figaxi,
              transform = ca.crs.PlateCarree(),
              facecolor = "None",
              edgecolor = "black",
              linewidth = 1.5,
              zorder = 4)
pre_lr_annual_wet_q95.plot(ax = figaxi,
             transform = ca.crs.PlateCarree(),
             cmap = cm.cm.lapaz_r,
             cbar_kwargs = {"label": "E contributing to P (mm)",
                            "shrink": 0.6,
                            "orientation": "horizontal",
                            "pad": 0.02},
             zorder = 3)
figaxi.set_title("")

quiver = ds_uv_wet.plot.quiver(ax = figaxi,
                     x = 'lon', y = 'lat', u = 'u', v = 'v', 
                     transform = ca.crs.PlateCarree(),
                     scale = 170, zorder = 5, alpha = 0.7)
mp.pyplot.quiverkey(quiver, 0.1, 0.1, 6.0, '6.0 m/s' ,labelpos = 'S')
mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures_v2", 
                               "Figure_5_Source_Region_LR_Wet_Q95.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )

#%% long rains (DRY)
figure = mp.pyplot.figure(figsize = (6,6))
figaxi = figure.add_subplot(1, 1, 1, 
                            projection = ca.crs.Geostationary(central_longitude = 48))
figaxi.set_global()
figaxi.add_feature(ca.feature.LAND, edgecolor='black', alpha = 1.0, zorder =1)
figaxi.add_feature(ca.feature.OCEAN, zorder = 0)
figaxi.coastlines(zorder = 2)
figaxi.add_feature(ca.feature.BORDERS, zorder = 1, linestyle = "--", color = "black",
                   linewidth = 0.5)
figaxi.gridlines(crs=ca.crs.PlateCarree(), color = "black",
                  zorder = 1, linestyle = "--")
shp_hoa.plot(ax = figaxi,
              transform = ca.crs.PlateCarree(),
              facecolor = "None",
              edgecolor = "black",
              linewidth = 1.5,
              zorder = 4)
pre_lr_annual_dry_q95.plot(ax = figaxi,
             transform = ca.crs.PlateCarree(),
             cmap = cm.cm.lapaz_r,
             cbar_kwargs = {"label": "E contributing to P (mm)",
                            "shrink": 0.6,
                            "orientation": "horizontal",
                            "pad": 0.02},
             zorder = 3)
figaxi.set_title("")

quiver = ds_uv_dry.plot.quiver(ax = figaxi,
                     x = 'lon', y = 'lat', u = 'u', v = 'v', 
                     transform = ca.crs.PlateCarree(),
                     scale = 170, zorder = 5, alpha = 0.7)
mp.pyplot.quiverkey(quiver, 0.1, 0.1, 6.0, '6.0 m/s' ,labelpos = 'S')

mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures_v2", 
                               "Figure_5_Source_Region_LR_Dry_Q95.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )

#%% anomalies (wet)
cmap = cm.cm.vik_r
norm = mp.colors.TwoSlopeNorm(vmin = -5, vcenter = 0.0, vmax = 70)
figure = mp.pyplot.figure(figsize = (6,6))
figaxi = figure.add_subplot(1, 1, 1, 
                            projection = ca.crs.Geostationary(central_longitude = 48))
figaxi.set_global()
figaxi.add_feature(ca.feature.LAND, edgecolor='black', alpha = 1.0, zorder =1)
figaxi.add_feature(ca.feature.OCEAN, zorder = 0)
figaxi.coastlines(zorder = 2)
figaxi.add_feature(ca.feature.BORDERS, zorder = 1, linestyle = "--", color = "black",
                   linewidth = 0.5)
figaxi.gridlines(crs=ca.crs.PlateCarree(), color = "black",
                  zorder = 1, linestyle = "--")
shp_hoa.plot(ax = figaxi,
              transform = ca.crs.PlateCarree(),
              facecolor = "None",
              edgecolor = "black",
              linewidth = 1.5,
              zorder = 4)
pre_lr_annual_wet_q95_anm.plot(ax = figaxi,
             transform = ca.crs.PlateCarree(),
             cmap = cm.cm.vik_r,
             norm = norm,
             cbar_kwargs = {"label": "E contributing to P (mm)",
                            "shrink": 0.6,
                            "orientation": "horizontal",
                            "pad": 0.02},
             zorder = 3)
figaxi.set_title("")
mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures_v2", 
                               "Figure_5_Source_Region_LR_Wet_Anom_Q95.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )

#%% anomalies (dry)
cmap = cm.cm.vik_r
norm = mp.colors.TwoSlopeNorm(vmin = -70, vcenter = 0.0, vmax = 5)
figure = mp.pyplot.figure(figsize = (6,6))
figaxi = figure.add_subplot(1, 1, 1, 
                            projection = ca.crs.Geostationary(central_longitude = 48))
figaxi.set_global()
figaxi.add_feature(ca.feature.LAND, edgecolor='black', alpha = 1.0, zorder =1)
figaxi.add_feature(ca.feature.OCEAN, zorder = 0)
figaxi.coastlines(zorder = 2)
figaxi.add_feature(ca.feature.BORDERS, zorder = 1, linestyle = "--", color = "black",
                   linewidth = 0.5)
figaxi.gridlines(crs=ca.crs.PlateCarree(), color = "black",
                  zorder = 1, linestyle = "--")
shp_hoa.plot(ax = figaxi,
              transform = ca.crs.PlateCarree(),
              facecolor = "None",
              edgecolor = "black",
              linewidth = 1.5,
              zorder = 4)
pre_lr_annual_dry_q95_anm.plot(ax = figaxi,
             transform = ca.crs.PlateCarree(),
             cmap = cm.cm.vik_r,
             norm = norm,
             cbar_kwargs = {"label": "E contributing to P (mm)",
                            "shrink": 0.6,
                            "orientation": "horizontal",
                            "pad": 0.02},
             zorder = 3)
figaxi.set_title("")
mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures_v2", 
                               "Figure_5_Source_Region_LR_Dry_Anom_Q95.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )

