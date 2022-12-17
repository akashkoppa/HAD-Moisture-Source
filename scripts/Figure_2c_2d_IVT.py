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

#%% read in the vimt data 
ds = xr.open_mfdataset("/Stor1/horn_of_africa/input/era5/long_rains/*.nc")
ds_ivt = ds["ivt"]

#%% read in the wind data 
ds_uv = ds[["qu","qv"]]

#%% average over time
ds_ivt = ds_ivt.mean(dim = "year")
ds_uv = ds_uv.mean(dim = "year")
ds_uv["qu"] = ds_uv["qu"]/ds_ivt
ds_uv["qv"] = ds_uv["qv"]/ds_ivt

lat_req = np.array(ds_uv.lat)
lat_req = lat_req[lat_req >= -35]
lat_req = lat_req[lat_req <= 35]
ds_uv = ds_uv.sel(lat = lat_req)

#%% resample the arrows
ds_uv_res = ds_uv.isel(lon = np.arange(0, 360, 3), lat = np.arange(0, 71, 3))

#%%
figure = mp.pyplot.figure(figsize = (6,6))
figaxi = figure.add_subplot(1, 1, 1, 
                            projection = ca.crs.Geostationary(central_longitude = 48))
figaxi.set_global()
figaxi.add_feature(ca.feature.LAND, edgecolor='black', alpha = 0.2, zorder =4)
figaxi.add_feature(ca.feature.OCEAN, zorder = 0)
figaxi.coastlines(zorder = 4)
figaxi.add_feature(ca.feature.BORDERS, zorder = 4, linestyle = "--", color = "black",
                   linewidth = 0.5)
figaxi.gridlines(crs=ca.crs.PlateCarree(), color = "black",
                  zorder = 1, linestyle = "--")
shp_hoa.plot(ax = figaxi,
              transform = ca.crs.PlateCarree(),
              facecolor = "None",
              edgecolor = "black",
              linewidth = 2.0,
              zorder = 5)
ds_ivt.plot(ax = figaxi,
             transform = ca.crs.PlateCarree(),
             cmap = cm.cm.lapaz_r,
             cbar_kwargs = {"label": "Integrated Water Vapour Transport (kg/m/s)",
                            "shrink": 0.6,
                            "orientation": "horizontal",
                            "pad": 0.02},
             zorder = 3)
quiver = ds_uv_res.plot.quiver(ax = figaxi,
                     x = 'lon', y = 'lat', u = 'qu', v = 'qv', 
                     transform = ca.crs.PlateCarree(),
                     alpha = 0.7,
                     scale = 40, zorder = 6)

mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures_v2", 
                               "Figure_3_IVT_LR.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )


