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
import cmcrameri as cm
import palettable as pl

#%% read in the hoa masks
shp_hoa = gp.read_file("/Stor1/horn_of_africa/input/study_region_masks/mask_hoa/mask_hoa.shp")

#%% read in the mask
maskfi = "/Stor1/horn_of_africa/input/study_region_masks/mask_hoa.nc"
maskda = xr.open_dataset(maskfi)
maskda = maskda["mask"]
#maskda = maskda.rio.set_crs(4326)

#%% read in the flexpart data 
flxfil = "/Stor1/horn_of_africa/input/hamster/drylands/daily"
ds = xr.open_mfdataset(os.path.join(flxfil, "*.nc"), chunks = {"lat": 50, "lon": 50, "time" :-1})

#%% calculate precipitation monthly climatologies
pre_clim = ds["E2P_EPs"].resample(time = "1M").sum()
pre_clim = pre_clim.groupby("time.month").mean()
pre_clim.compute()
pre_clim_mean = pre_clim.sum(dim = ["lat","lon"]).to_pandas()
pre_clim_mean = pre_clim_mean/np.array(maskda.where(maskda == 1).count())

#%% calculate the precipitation monthly standard deviations
pre_clim = ds["E2P_EPs"].resample(time = "1M").sum()
pre_clim_mon = pre_clim.sum(dim = ["lat", "lon"]).to_pandas()
pre_clim_mon = pre_clim_mon/np.array(maskda.where(maskda == 1).count())
pre_clim_std = pre_clim_mon.groupby(pre_clim_mon.index.month).std()
pre_clim_std = pre_clim_std/2
#pre_clim_std = pre_clim_std * 2


#%% get required number of colors
cmap = mp.colors.ListedColormap(pl.scientific.diverging.Vik_10.mpl_colors, N = 10)
cmap_1 = mp.colors.ListedColormap(pl.scientific.sequential.GrayC_10_r.mpl_colors, N = 10)
cmap_2 = mp.colors.ListedColormap(pl.scientific.diverging.Cork_10.mpl_colors, N = 10)


#%% plot the seasonal cycle plot
pre_clim_mean.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
pre_clim_mean.name = "P"
pre_clim_std.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
pre_clim_std.name = "SD"
pre_clim_final = pd.concat([pre_clim_mean, pre_clim_std], axis = 1)
colors = ["black", "black", 
          "#5e81ac","#5e81ac","#5e81ac",
          "black", "black", "black", "black",
          "#81a1c1","#81a1c1","#81a1c1"]

colors = [cmap_1(5), cmap_1(5), 
          cmap(2),cmap(2),cmap(2),
          cmap_1(5), cmap_1(5), cmap_1(5), cmap_1(5),
          cmap_2(8),cmap_2(8),cmap_2(8)]

hatches = ["","","", "","","","","","","","",""]

figure = mp.pyplot.figure(figsize = (7,4))
figaxi = figure.add_subplot(1, 1, 1)
pre_clim_final.plot.bar(ax = figaxi, y = "P", color = colors, stacked = True,
                        edgecolor = "black", linewidth = 0.7)
bars = figaxi.patches
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)
figaxi.set_xlabel("Month", family = "Arial")
figaxi.set_ylabel("Precipitation (mm)")
figaxi.errorbar(pre_clim_final.index, pre_clim_final["P"], 
                yerr = pre_clim_final["SD"],
                linewidth = 1.5, fmt = " ", color = "black", capsize = 4)
figaxi.get_legend().remove()
mp.pyplot.savefig("/Stor1/horn_of_africa/hoa_paper/figures_v3/Figure_1_Seasonal_Cycle.png",
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600)




