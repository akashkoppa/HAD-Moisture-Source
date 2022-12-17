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

#%% calculate precipitation totals 
# long rain
pre_month = ds["E2P_EPs"].resample(time = "1M").sum()
pre_lr = pre_month.sel(time = pre_month.time.dt.month.isin([3,4,5]))
pre_lr = pre_lr.groupby("time.year").sum()
pre_lr = pre_lr.mean(dim = ["year"])

# March
pre_mar = pre_month.sel(time = pre_month.time.dt.month.isin([3]))
pre_mar = pre_mar.mean(dim = ["time"])

# April
pre_apr = pre_month.sel(time = pre_month.time.dt.month.isin([4]))
pre_apr = pre_apr.mean(dim = ["time"])

# May
pre_may = pre_month.sel(time = pre_month.time.dt.month.isin([5]))
pre_may = pre_may.mean(dim = ["time"])

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

#%% plot the sources regions of long and short rains
# long rains
pre_lr_plot = xr.where(q95_mask_lr == 1.0, pre_lr, np.nan)
#%%
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
              zorder = 4)

mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures_v2", 
                               "Figure_3_Source_Region_LR.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )

#%% short rains
pre_sr_plot = xr.where(q95_mask_sr == 1.0, pre_sr, np.nan)
pre_sr
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
              zorder = 4)
mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures_v2", 
                               "Figure_3_Source_Region_SR.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )


