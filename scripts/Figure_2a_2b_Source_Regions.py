#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Figure 1: Study Area
-------------------------------------------------------------------------------
Author: Akash Koppa
Date: 2022-02-02

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
#maskda = maskda.rio.set_crs(4326)

#%% read in the hoa masks
shp_hoa = gp.read_file("/Stor1/horn_of_africa/input/study_region_masks/mask_hoa/mask_hoa.shp")
shp_q50_lr = gp.read_file("/Stor1/horn_of_africa/hoa_paper/mask_source_region_shp/pre_lr_q50_mask.shp")
shp_q50_sr = gp.read_file("/Stor1/horn_of_africa/hoa_paper/mask_source_region_shp/pre_sr_q50_mask.shp")

#%% read in the Q95 mask
q95_mask_lr = xr.open_rasterio("/Stor1/horn_of_africa/hoa_paper/mask_source_region/pre_lr_q95_mask.tif") 
q95_mask_sr = xr.open_rasterio("/Stor1/horn_of_africa/hoa_paper/mask_source_region/pre_sr_q95_mask.tif") 

q95_mask_lr = q95_mask_lr.isel(band = 0).drop("band").rename({"x":"lon", "y":"lat"})
q95_mask_sr = q95_mask_sr.isel(band = 0).drop("band").rename({"x":"lon", "y":"lat"})

#%% read in the flexpart data 
flxfil = "/Stor1/horn_of_africa/input/hamster/drylands/daily"
ds = xr.open_mfdataset(os.path.join(flxfil, "*.nc"), chunks = {"lat": 50, "lon": 50, "time" :-1})

#%% read in the wind data 
#ds_uv = xr.open_dataset("/Stor1/horn_of_africa/input/era5/input/u_v_monthly.nc")
#ds_uv = ds_uv.sel(expver = 1)

#%% regrid the wind data to the flexpart grid
#regridder = xe.Regridder(ds_uv, maskda, "nearest_s2d")
#ds_uv = regridder(ds_uv)

#%% calculate precipitation totals 
# long rain
pre_month = ds["E2P_EPs"].resample(time = "1M").sum()
pre_lr = pre_month.sel(time = pre_month.time.dt.month.isin([3,4,5]))
pre_lr = pre_lr.groupby("time.year").sum()
pre_lr = pre_lr.mean(dim = ["year"])

# March
pre_mar = pre_month.sel(time = pre_month.time.dt.month.isin([3]))
pre_mar = pre_mar.mean(dim = ["time"])
#pre_mar = pre_mar.groupby("time.year").sum(dim = ["lat", "lon"]).to_pandas()
#pre_mar = pre_mar/np.array(maskda.where(maskda == 1).count())

# April
pre_apr = pre_month.sel(time = pre_month.time.dt.month.isin([4]))
pre_apr = pre_apr.mean(dim = ["time"])

# May
pre_may = pre_month.sel(time = pre_month.time.dt.month.isin([5]))
pre_may = pre_may.mean(dim = ["time"])

# select wind data for the long rains
#ds_uv_lr = ds_uv.sel(time = ds_uv.time.dt.month.isin([3,4,5]))
#ds_uv_lr.compute()
#ds_uv_lr = ds_uv_lr.groupby("time.year").mean()
#ds_uv_lr = ds_uv_lr.mean(dim = ["year"])
#ds_uv_lr = xr.where(pre_lr < 0.5, np.nan, ds_uv_lr)
# resample to have less number of arrows
#lat_req = np.array(ds_uv_lr.lat)
#lat_req = lat_req[lat_req >= -35]
#lat_req = lat_req[lat_req <= 35]
#ds_uv_lr = ds_uv_lr.sel(lat = lat_req)
#ds_uv_lr = ds_uv_lr.isel(lon = np.arange(0, 360, 5), lat = np.arange(0, 71, 5))
#ds_uv_lr = ds_uv_lr.drop("expver")
#%% short rain
pre_sr = pre_month.sel(time = pre_month.time.dt.month.isin([10,11,12]))
pre_sr = pre_sr.groupby("time.year").sum()
pre_sr = pre_sr.mean(dim = ["year"])

# October
pre_oct = pre_month.sel(time = pre_month.time.dt.month.isin([10]))
pre_oct = pre_oct.mean(dim = ["time"])

# November
pre_nov = pre_month.sel(time = pre_month.time.dt.month.isin([11]))
pre_nov = pre_nov.mean(dim = ["time"])

# December
pre_dec = pre_month.sel(time = pre_month.time.dt.month.isin([12]))
pre_dec = pre_dec.mean(dim = ["time"])

# select wind data for the long rains
#ds_uv_sr = ds_uv.sel(time = ds_uv.time.dt.month.isin([10,11,12]))
#ds_uv_sr.compute()
#ds_uv_sr = ds_uv_sr.groupby("time.year").mean()
#ds_uv_sr = ds_uv_sr.mean(dim = ["year"])
#ds_uv_sr = xr.where(pre_sr < 0.5, np.nan, ds_uv_sr)
## resample to have less number of arrows
#ds_uv_sr = ds_uv_sr.isel(lon = np.arange(0, 360, 5), lat = np.arange(0, 181, 5))
#ds_uv_sr = ds_uv_sr.drop("expver")

#%% plot the sources regions of long and short rains
# long rains
pre_lr_plot = xr.where(q95_mask_lr == 1.0, pre_lr, np.nan)
#%%
figure = mp.pyplot.figure(figsize = (6,6))
figaxi = figure.add_subplot(1, 1, 1, 
                            projection = ca.crs.Geostationary(central_longitude = 48))
                            #projection = ca.crs.Mercator())
figaxi.set_global()
#figaxi.set_extent([-120, 150, -36, 36])
#figaxi.stock_img()
figaxi.add_feature(ca.feature.LAND, edgecolor='black', alpha = 1.0, zorder =1)
#figaxi.add_feature(ca.feature.NaturalEarthFeature('physical', 'land', scale='50m'))
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
              linewidth = 2.0,
              zorder = 4)
pre_lr_plot.plot(ax = figaxi,
             transform = ca.crs.PlateCarree(),
             cmap = cm.cm.lapaz_r,
             cbar_kwargs = {"label": "E contributing to P (mm/month)",
                            "shrink": 0.6,
                            "orientation": "horizontal",
                            "pad": 0.02},
             zorder = 3)
shp_q50_lr.plot(ax = figaxi,
              transform = ca.crs.PlateCarree(),
              facecolor = "None",
              edgecolor = "red",
              linewidth = 0.75,
              #linestyle = "dashed",
              zorder = 4)
#quiver = ds_uv_lr.plot.quiver(ax = figaxi,
#                     x = 'lon', y = 'lat', u = 'u', v = 'v', 
#                     transform = ca.crs.PlateCarree(),
#                     scale = 100, zorder = 5, )
#mp.pyplot.quiverkey(quiver, 0.1, 0.1, 6.0, '6.0 m/s' ,labelpos = 'S')

#figure.patch.set_alpha(0)
mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures_v2", 
                               "Figure_3_Source_Region_LR.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )

#%% short rains
#pre_sr_plot = xr.where(pre_sr < 0.5, np.nan, pre_sr)
pre_sr_plot = xr.where(q95_mask_sr == 1.0, pre_sr, np.nan)
pre_sr
figure = mp.pyplot.figure(figsize = (6,6))
figaxi = figure.add_subplot(1, 1, 1, 
                            projection = ca.crs.Geostationary(central_longitude = 48))
figaxi.set_global()
#figaxi.set_extent(-60, 60, -)
#figaxi.stock_img()
figaxi.add_feature(ca.feature.LAND, edgecolor='black', alpha = 1.0, zorder =1)
#figaxi.add_feature(ca.feature.NaturalEarthFeature('physical', 'land', scale='50m'))
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
              linewidth = 2.0,
              zorder = 4)
pre_sr_plot.plot(ax = figaxi,
             transform = ca.crs.PlateCarree(),
             cmap = cm.cm.lapaz_r,
             cbar_kwargs = {"label": "E contributing to P (mm/month)",
                            "shrink": 0.6,
                            "orientation": "horizontal",
                            "pad": 0.02},
             zorder = 3)
shp_q50_sr.plot(ax = figaxi,
              transform = ca.crs.PlateCarree(),
              facecolor = "None",
              edgecolor = "red",
              linewidth = 0.75,
              #linestyle = "dashed",
              zorder = 4)
#figure.patch.set_alpha(0)
mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures_v2", 
                               "Figure_3_Source_Region_SR.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )
"""
#%% plot the source regions
# March
pre_mar_plot = xr.where(pre_mar < 0.5, np.nan, pre_mar)
figure = mp.pyplot.figure(figsize = (6,6))
figaxi = figure.add_subplot(1, 1, 1, 
                            projection = ca.crs.Geostationary(central_longitude = 48))
figaxi.set_global()
#figaxi.set_extent(-60, 60, -)
#figaxi.stock_img()
figaxi.add_feature(ca.feature.LAND, edgecolor='black', alpha = 1.0, zorder =1)
#figaxi.add_feature(ca.feature.NaturalEarthFeature('physical', 'land', scale='50m'))
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
              linewidth = 2.0,
              zorder = 4)
pre_mar_plot.plot(ax = figaxi,
             transform = ca.crs.PlateCarree(),
             cmap = cm.cm.batlowW_r,
             cbar_kwargs = {"label": "E contributing to P (mm/month)",
                            "shrink": 0.6,
                            "orientation": "horizontal",
                            "pad": 0.02},
             zorder = 3)
#figure.patch.set_alpha(0)
mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures", 
                               "Figure_3_Source_Region_Mar.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )

# April
pre_apr_plot = xr.where(pre_apr < 0.5, np.nan, pre_apr)
figure = mp.pyplot.figure(figsize = (6,6))
figaxi = figure.add_subplot(1, 1, 1, 
                            projection = ca.crs.Geostationary(central_longitude = 48))
figaxi.set_global()
#figaxi.set_extent(-60, 60, -)
#figaxi.stock_img()
figaxi.add_feature(ca.feature.LAND, edgecolor='black', alpha = 1.0, zorder =1)
#figaxi.add_feature(ca.feature.NaturalEarthFeature('physical', 'land', scale='50m'))
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
              linewidth = 2.0,
              zorder = 4)
pre_apr_plot.plot(ax = figaxi,
             transform = ca.crs.PlateCarree(),
             cmap = cm.cm.batlowW_r,
             cbar_kwargs = {"label": "E contributing to P (mm/month)",
                            "shrink": 0.6,
                            "orientation": "horizontal",
                            "pad": 0.02},
             zorder = 3)
#figure.patch.set_alpha(0)
mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures", 
                               "Figure_3_Source_Region_Apr.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )

# May
pre_may_plot = xr.where(pre_may < 0.5, np.nan, pre_mar)
figure = mp.pyplot.figure(figsize = (6,6))
figaxi = figure.add_subplot(1, 1, 1, 
                            projection = ca.crs.Geostationary(central_longitude = 48))
figaxi.set_global()
#figaxi.set_extent(-60, 60, -)
#figaxi.stock_img()
figaxi.add_feature(ca.feature.LAND, edgecolor='black', alpha = 1.0, zorder =1)
#figaxi.add_feature(ca.feature.NaturalEarthFeature('physical', 'land', scale='50m'))
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
              linewidth = 2.0,
              zorder = 4)
pre_may_plot.plot(ax = figaxi,
             transform = ca.crs.PlateCarree(),
             cmap = cm.cm.batlowW_r,
             cbar_kwargs = {"label": "E contributing to P (mm/month)",
                            "shrink": 0.6,
                            "orientation": "horizontal",
                            "pad": 0.02},
             zorder = 3)
#figure.patch.set_alpha(0)
mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures", 
                               "Figure_3_Source_Region_May.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )

#%% Short rain
# October
pre_oct_plot = xr.where(pre_oct < 0.5, np.nan, pre_oct)
figure = mp.pyplot.figure(figsize = (6,6))
figaxi = figure.add_subplot(1, 1, 1, 
                            projection = ca.crs.Geostationary(central_longitude = 48))
figaxi.set_global()
#figaxi.set_extent(-60, 60, -)
#figaxi.stock_img()
figaxi.add_feature(ca.feature.LAND, edgecolor='black', alpha = 1.0, zorder =1)
#figaxi.add_feature(ca.feature.NaturalEarthFeature('physical', 'land', scale='50m'))
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
              linewidth = 2.0,
              zorder = 4)
pre_oct_plot.plot(ax = figaxi,
             transform = ca.crs.PlateCarree(),
             cmap = cm.cm.batlowW_r,
             cbar_kwargs = {"label": "E contributing to P (mm/month)",
                            "shrink": 0.6,
                            "orientation": "horizontal",
                            "pad": 0.02},
             zorder = 3)
#figure.patch.set_alpha(0)
mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures", 
                               "Figure_3_Source_Region_Oct.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )

# November
pre_nov_plot = xr.where(pre_apr < 0.5, np.nan, pre_nov)
figure = mp.pyplot.figure(figsize = (6,6))
figaxi = figure.add_subplot(1, 1, 1, 
                            projection = ca.crs.Geostationary(central_longitude = 48))
figaxi.set_global()
#figaxi.set_extent(-60, 60, -)
#figaxi.stock_img()
figaxi.add_feature(ca.feature.LAND, edgecolor='black', alpha = 1.0, zorder =1)
#figaxi.add_feature(ca.feature.NaturalEarthFeature('physical', 'land', scale='50m'))
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
              linewidth = 2.0,
              zorder = 4)
pre_nov_plot.plot(ax = figaxi,
             transform = ca.crs.PlateCarree(),
             cmap = cm.cm.batlowW_r,
             cbar_kwargs = {"label": "E contributing to P (mm/month)",
                            "shrink": 0.6,
                            "orientation": "horizontal",
                            "pad": 0.02},
             zorder = 3)
#figure.patch.set_alpha(0)
mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures", 
                               "Figure_3_Source_Region_Nov.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )

# December
pre_dec_plot = xr.where(pre_may < 0.5, np.nan, pre_dec)
figure = mp.pyplot.figure(figsize = (6,6))
figaxi = figure.add_subplot(1, 1, 1, 
                            projection = ca.crs.Geostationary(central_longitude = 48))
figaxi.set_global()
#figaxi.set_extent(-60, 60, -)
#figaxi.stock_img()
figaxi.add_feature(ca.feature.LAND, edgecolor='black', alpha = 1.0, zorder =1)
#figaxi.add_feature(ca.feature.NaturalEarthFeature('physical', 'land', scale='50m'))
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
              linewidth = 2.0,
              zorder = 4)
pre_dec_plot.plot(ax = figaxi,
             transform = ca.crs.PlateCarree(),
             cmap = cm.cm.batlowW_r,
             cbar_kwargs = {"label": "E contributing to P (mm/month)",
                            "shrink": 0.6,
                            "orientation": "horizontal",
                            "pad": 0.02},
             zorder = 3)
#figure.patch.set_alpha(0)
mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures", 
                               "Figure_3_Source_Region_Dec.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )
"""


