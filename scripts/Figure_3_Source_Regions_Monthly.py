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

shp_q50_mar = gp.read_file("/Stor1/horn_of_africa/hoa_paper/mask_source_region_shp/pre_mar_q50_mask.shp")
shp_q50_apr = gp.read_file("/Stor1/horn_of_africa/hoa_paper/mask_source_region_shp/pre_apr_q50_mask.shp")
shp_q50_may = gp.read_file("/Stor1/horn_of_africa/hoa_paper/mask_source_region_shp/pre_may_q50_mask.shp")
shp_q50_oct = gp.read_file("/Stor1/horn_of_africa/hoa_paper/mask_source_region_shp/pre_oct_q50_mask.shp")
shp_q50_nov = gp.read_file("/Stor1/horn_of_africa/hoa_paper/mask_source_region_shp/pre_nov_q50_mask.shp")
shp_q50_dec = gp.read_file("/Stor1/horn_of_africa/hoa_paper/mask_source_region_shp/pre_dec_q50_mask.shp")

#%% read in the q95 mask
q95_mask_mar = xr.open_rasterio("/Stor1/horn_of_africa/hoa_paper/mask_source_region/pre_mar_q95_mask.tif") 
q95_mask_apr = xr.open_rasterio("/Stor1/horn_of_africa/hoa_paper/mask_source_region/pre_apr_q95_mask.tif") 
q95_mask_may = xr.open_rasterio("/Stor1/horn_of_africa/hoa_paper/mask_source_region/pre_may_q95_mask.tif") 

q95_mask_mar = q95_mask_mar.isel(band = 0).drop("band").rename({"x":"lon", "y":"lat"})
q95_mask_apr = q95_mask_apr.isel(band = 0).drop("band").rename({"x":"lon", "y":"lat"})
q95_mask_may = q95_mask_may.isel(band = 0).drop("band").rename({"x":"lon", "y":"lat"})

q95_mask_oct = xr.open_rasterio("/Stor1/horn_of_africa/hoa_paper/mask_source_region/pre_oct_q95_mask.tif") 
q95_mask_nov = xr.open_rasterio("/Stor1/horn_of_africa/hoa_paper/mask_source_region/pre_nov_q95_mask.tif") 
q95_mask_dec = xr.open_rasterio("/Stor1/horn_of_africa/hoa_paper/mask_source_region/pre_dec_q95_mask.tif") 

q95_mask_oct = q95_mask_oct.isel(band = 0).drop("band").rename({"x":"lon", "y":"lat"})
q95_mask_nov = q95_mask_nov.isel(band = 0).drop("band").rename({"x":"lon", "y":"lat"})
q95_mask_dec = q95_mask_dec.isel(band = 0).drop("band").rename({"x":"lon", "y":"lat"})

#%% read in the hoa masks
shp_hoa = gp.read_file("/Stor1/horn_of_africa/input/study_region_masks/mask_hoa/mask_hoa.shp")

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

# select wind data for the long rains
# March
ds_uv_mar = ds_uv.sel(time = ds_uv.time.dt.month.isin([3]))
ds_uv_mar.compute()
ds_uv_mar = ds_uv_mar.groupby("time.year").mean()
ds_uv_mar = ds_uv_mar.mean(dim = ["year"])
# resample to have less number of arrows
lat_req = np.array(ds_uv_mar.lat)
lat_req = lat_req[lat_req >= -35]
lat_req = lat_req[lat_req <= 35]
ds_uv_mar = ds_uv_mar.sel(lat = lat_req)
ds_uv_mar = ds_uv_mar.isel(lon = np.arange(0, 360, 3), lat = np.arange(0, 71, 3))
ds_uv_mar = ds_uv_mar.drop("expver")

# April
ds_uv_apr = ds_uv.sel(time = ds_uv.time.dt.month.isin([4]))
ds_uv_apr.compute()
ds_uv_apr = ds_uv_apr.groupby("time.year").mean()
ds_uv_apr = ds_uv_apr.mean(dim = ["year"])
# resample to have less number of arrows
lat_req = np.array(ds_uv_apr.lat)
lat_req = lat_req[lat_req >= -35]
lat_req = lat_req[lat_req <= 35]
ds_uv_apr = ds_uv_apr.sel(lat = lat_req)
ds_uv_apr = ds_uv_apr.isel(lon = np.arange(0, 360, 3), lat = np.arange(0, 71, 3))
ds_uv_apr = ds_uv_apr.drop("expver")

# May
ds_uv_may = ds_uv.sel(time = ds_uv.time.dt.month.isin([5]))
ds_uv_may.compute()
ds_uv_may = ds_uv_may.groupby("time.year").mean()
ds_uv_may = ds_uv_may.mean(dim = ["year"])
# resample to have less number of arrows
lat_req = np.array(ds_uv_may.lat)
lat_req = lat_req[lat_req >= -35]
lat_req = lat_req[lat_req <= 35]
ds_uv_may = ds_uv_may.sel(lat = lat_req)
ds_uv_may = ds_uv_may.isel(lon = np.arange(0, 360, 3), lat = np.arange(0, 71, 3))
ds_uv_may = ds_uv_may.drop("expver")

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

# October
ds_uv_oct = ds_uv.sel(time = ds_uv.time.dt.month.isin([10]))
ds_uv_oct.compute()
ds_uv_oct = ds_uv_oct.groupby("time.year").mean()
ds_uv_oct = ds_uv_oct.mean(dim = ["year"])
# resample to have less number of arrows
lat_req = np.array(ds_uv_oct.lat)
lat_req = lat_req[lat_req >= -35]
lat_req = lat_req[lat_req <= 35]
ds_uv_oct = ds_uv_oct.sel(lat = lat_req)
ds_uv_oct = ds_uv_oct.isel(lon = np.arange(0, 360, 3), lat = np.arange(0, 71, 3))
ds_uv_oct = ds_uv_oct.drop("expver")

# November
ds_uv_nov = ds_uv.sel(time = ds_uv.time.dt.month.isin([11]))
ds_uv_nov.compute()
ds_uv_nov = ds_uv_nov.groupby("time.year").mean()
ds_uv_nov = ds_uv_nov.mean(dim = ["year"])
# resample to have less number of arrows
lat_req = np.array(ds_uv_nov.lat)
lat_req = lat_req[lat_req >= -35]
lat_req = lat_req[lat_req <= 35]
ds_uv_nov = ds_uv_nov.sel(lat = lat_req)
ds_uv_nov = ds_uv_nov.isel(lon = np.arange(0, 360, 3), lat = np.arange(0, 71, 3))
ds_uv_nov = ds_uv_nov.drop("expver")

# December
ds_uv_dec = ds_uv.sel(time = ds_uv.time.dt.month.isin([12]))
ds_uv_dec.compute()
ds_uv_dec = ds_uv_dec.groupby("time.year").mean()
ds_uv_dec = ds_uv_dec.mean(dim = ["year"])
# resample to have less number of arrows
lat_req = np.array(ds_uv_dec.lat)
lat_req = lat_req[lat_req >= -35]
lat_req = lat_req[lat_req <= 35]
ds_uv_dec = ds_uv_dec.sel(lat = lat_req)
ds_uv_dec = ds_uv_dec.isel(lon = np.arange(0, 360, 3), lat = np.arange(0, 71, 3))
ds_uv_dec = ds_uv_dec.drop("expver")

#%% March
pre_mar_plot = xr.where(q95_mask_mar == 1.0, pre_mar, np.nan)
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
pre_mar_plot.plot(ax = figaxi,
             transform = ca.crs.PlateCarree(),
             cmap = cm.cm.lapaz_r,
             cbar_kwargs = {"label": "E contributing to P (mm/month)",
                            "shrink": 0.6,
                            "orientation": "horizontal",
                            "pad": 0.02},
             zorder = 3)
shp_q50_mar.plot(ax = figaxi,
              transform = ca.crs.PlateCarree(),
              facecolor = "None",
              edgecolor = "red",
              linewidth = 0.75,
              zorder = 4)

quiver = ds_uv_mar.plot.quiver(ax = figaxi,
                     x = 'lon', y = 'lat', u = 'u', v = 'v', 
                     transform = ca.crs.PlateCarree(),
                     scale = 170, zorder = 5, alpha = 0.7)
mp.pyplot.quiverkey(quiver, 0.1, 0.1, 6.0, '6.0 m/s' ,labelpos = 'S')
#figure.patch.set_alpha(0)
mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures_v2", 
                               "Figure_3_Source_Region_Mar.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )

#%% April
pre_apr_plot = xr.where(q95_mask_apr == 1.0, pre_apr, np.nan)
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
pre_apr_plot.plot(ax = figaxi,
             transform = ca.crs.PlateCarree(),
             cmap = cm.cm.lapaz_r,
             cbar_kwargs = {"label": "E contributing to P (mm/month)",
                            "shrink": 0.6,
                            "orientation": "horizontal",
                            "pad": 0.02},
             zorder = 3)
shp_q50_apr.plot(ax = figaxi,
              transform = ca.crs.PlateCarree(),
              facecolor = "None",
              edgecolor = "red",
              linewidth = 0.75,
              zorder = 4)

quiver = ds_uv_apr.plot.quiver(ax = figaxi,
                     x = 'lon', y = 'lat', u = 'u', v = 'v', 
                     transform = ca.crs.PlateCarree(),
                     scale = 170, zorder = 5, alpha = 0.7)
mp.pyplot.quiverkey(quiver, 0.1, 0.1, 6.0, '6.0 m/s' ,labelpos = 'S')
mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures_v2", 
                               "Figure_3_Source_Region_Apr.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )

#%% May
pre_may_plot = xr.where(q95_mask_may == 1.0, pre_may, np.nan)
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
pre_may_plot.plot(ax = figaxi,
             transform = ca.crs.PlateCarree(),
             cmap = cm.cm.lapaz_r,
             cbar_kwargs = {"label": "E contributing to P (mm/month)",
                            "shrink": 0.6,
                            "orientation": "horizontal",
                            "pad": 0.02},
             zorder = 3)
shp_q50_may.plot(ax = figaxi,
              transform = ca.crs.PlateCarree(),
              facecolor = "None",
              edgecolor = "red",
              linewidth = 0.75,
              zorder = 4)

quiver = ds_uv_may.plot.quiver(ax = figaxi,
                     x = 'lon', y = 'lat', u = 'u', v = 'v', 
                     transform = ca.crs.PlateCarree(),
                     scale = 170, zorder = 5, alpha = 0.7)
mp.pyplot.quiverkey(quiver, 0.1, 0.1, 6.0, '6.0 m/s' ,labelpos = 'S')
mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures_v2", 
                               "Figure_3_Source_Region_May.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )

#%% Oct
pre_oct_plot = xr.where(q95_mask_oct == 1.0, pre_oct, np.nan)
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
pre_oct_plot.plot(ax = figaxi,
             transform = ca.crs.PlateCarree(),
             cmap = cm.cm.lapaz_r,
             cbar_kwargs = {"label": "E contributing to P (mm/month)",
                            "shrink": 0.6,
                            "orientation": "horizontal",
                            "pad": 0.02},
             zorder = 3)
shp_q50_oct.plot(ax = figaxi,
              transform = ca.crs.PlateCarree(),
              facecolor = "None",
              edgecolor = "red",
              linewidth = 0.75,
              zorder = 4)

quiver = ds_uv_oct.plot.quiver(ax = figaxi,
                     x = 'lon', y = 'lat', u = 'u', v = 'v', 
                     transform = ca.crs.PlateCarree(),
                     scale = 170, zorder = 5, alpha = 0.7)
mp.pyplot.quiverkey(quiver, 0.1, 0.1, 6.0, '6.0 m/s' ,labelpos = 'S')
mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures_v2", 
                               "Figure_3_Source_Region_Oct.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )

#%% Nov
pre_nov_plot = xr.where(q95_mask_nov == 1.0, pre_nov, np.nan)
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
pre_nov_plot.plot(ax = figaxi,
             transform = ca.crs.PlateCarree(),
             cmap = cm.cm.lapaz_r,
             cbar_kwargs = {"label": "E contributing to P (mm/month)",
                            "shrink": 0.6,
                            "orientation": "horizontal",
                            "pad": 0.02},
             zorder = 3)
shp_q50_nov.plot(ax = figaxi,
              transform = ca.crs.PlateCarree(),
              facecolor = "None",
              edgecolor = "red",
              linewidth = 0.75,
              zorder = 4)

quiver = ds_uv_nov.plot.quiver(ax = figaxi,
                     x = 'lon', y = 'lat', u = 'u', v = 'v', 
                     transform = ca.crs.PlateCarree(),
                     scale = 170, zorder = 5, alpha = 0.7)
mp.pyplot.quiverkey(quiver, 0.1, 0.1, 6.0, '6.0 m/s' ,labelpos = 'S')
mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures_v2", 
                               "Figure_3_Source_Region_Nov.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )

#%% Dec
pre_dec_plot = xr.where(q95_mask_dec == 1.0, pre_dec, np.nan)
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
pre_dec_plot.plot(ax = figaxi,
             transform = ca.crs.PlateCarree(),
             cmap = cm.cm.lapaz_r,
             cbar_kwargs = {"label": "E contributing to P (mm/month)",
                            "shrink": 0.6,
                            "orientation": "horizontal",
                            "pad": 0.02},
             zorder = 3)
shp_q50_dec.plot(ax = figaxi,
              transform = ca.crs.PlateCarree(),
              facecolor = "None",
              edgecolor = "red",
              linewidth = 0.75,
              zorder = 4)

quiver = ds_uv_dec.plot.quiver(ax = figaxi,
                     x = 'lon', y = 'lat', u = 'u', v = 'v', 
                     transform = ca.crs.PlateCarree(),
                     scale = 170, zorder = 5, alpha = 0.7)
mp.pyplot.quiverkey(quiver, 0.1, 0.1, 6.0, '6.0 m/s' ,labelpos = 'S')
mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures_v2", 
                               "Figure_3_Source_Region_Dec.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )

