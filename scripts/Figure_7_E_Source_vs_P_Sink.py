#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Akash Koppa

"""
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
import xesmf as xesmf

#%% read in the mask
maskfi = "/Stor1/horn_of_africa/input/study_region_masks/mask_hoa.nc"
maskda = xr.open_dataset(maskfi)
maskda = maskda["mask"]

#%% derive the precipitation estimates
# read in the flexpart data 
flxfil = "/Stor1/horn_of_africa/input/hamster/drylands/daily"
ds = xr.open_mfdataset(os.path.join(flxfil, "*.nc"), chunks = {"lat": 50, "lon": 50, "time" :-1})

#%% calculate evaporation from ocean
ds_e = xr.open_dataset("/Stor1/horn_of_africa/input/E/monthly/E_GLEAM_OAFLUX_1deg_monthly_1980-2016.nc")
ds_e = ds_e["evaporation"]

#%% loop through the years and calculate the evaporation from source regions
years = np.unique(np.array(ds_e.time.dt.year))
e_lr_tmp_final = []
e_sr_tmp_final = []
for q in [50, 55, 65, 75, 85, 90, 95]:
    q_reqd = "q" + str(q)
    print(q_reqd)
    
    e_lr_tmp = []
    e_sr_tmp = []
    
    for year in years:
        print(year)
        ds_e_tmp = ds_e.sel(time = ds_e.time.dt.year.isin([year]))
        # read in the Q95 mask
        q95_mask_lr = xr.open_rasterio("/Stor1/horn_of_africa/hoa_paper/mask_source_region_annual/lr/pre_lr_" + q_reqd + "_mask_" + str(year) + ".tif") 
        q95_mask_sr = xr.open_rasterio("/Stor1/horn_of_africa/hoa_paper/mask_source_region_annual/sr/pre_sr_" + q_reqd + "_mask_" + str(year) + ".tif") 

        q95_mask_lr = q95_mask_lr.isel(band = 0).drop("band").rename({"x":"lon", "y":"lat"})
        q95_mask_sr = q95_mask_sr.isel(band = 0).drop("band").rename({"x":"lon", "y":"lat"})

        #
        e_lr = xr.where(q95_mask_lr == 1, ds_e_tmp, np.nan)
        e_sr = xr.where(q95_mask_sr == 1, ds_e_tmp, np.nan)

        # long rains
        e_lr = e_lr.mean(dim = ["lat","lon"]).to_pandas()
        e_lr = e_lr.loc[e_lr.index.month.isin([2,3,4,5])]
        e_lr = e_lr.resample("1Y").mean()
        e_lr.index = e_lr.index.year
        e_lr_tmp.append(e_lr)
        
        # short rains
        e_sr = e_sr.mean(dim = ["lat","lon"]).to_pandas()
        e_sr = e_sr.loc[e_sr.index.month.isin([9,10,11,12])]
        e_sr = e_sr.resample("1Y").mean()
        e_sr.index = e_sr.index.year
        e_sr_tmp.append(e_sr)
    
    e_lr = pd.concat(e_lr_tmp, axis = 0)
    e_lr.name = q_reqd
    e_sr = pd.concat(e_sr_tmp, axis = 0)
    e_sr.name = q_reqd
    
    e_lr_tmp_final.append(e_lr)
    e_sr_tmp_final.append(e_sr)
    
#%%
e_lr = pd.concat(e_lr_tmp_final, axis = 1)
e_sr = pd.concat(e_sr_tmp_final, axis = 1)

#%% loop through the years and calculate the weighted evaporation from source regions
years = np.unique(np.array(ds_e.time.dt.year))
e_lr_weight_tmp = []
e_sr_weight_tmp = []
for year in years:
    print(year)
    ds_e_tmp = ds_e.sel(time = ds_e.time.dt.year.isin([year]))
    # read in the Q95 mask
    q95_mask_lr = xr.open_rasterio("/Stor1/horn_of_africa/hoa_paper/mask_source_region_annual/lr/pre_lr_q95_mask_" + str(year) + ".tif") 
    q95_mask_sr = xr.open_rasterio("/Stor1/horn_of_africa/hoa_paper/mask_source_region_annual/sr/pre_sr_q95_mask_" + str(year) + ".tif") 

    q95_mask_lr = q95_mask_lr.isel(band = 0).drop("band").rename({"x":"lon", "y":"lat"})
    q95_mask_sr = q95_mask_sr.isel(band = 0).drop("band").rename({"x":"lon", "y":"lat"})
    
    # read in the absolute value rasters
    q95_absolute_lr = xr.open_rasterio("/Stor1/horn_of_africa/hoa_paper/mask_source_region_annual/lr/pre_lr_q95_absolute_" + str(year) + ".tif") 
    q95_absolute_sr = xr.open_rasterio("/Stor1/horn_of_africa/hoa_paper/mask_source_region_annual/sr/pre_sr_q95_absolute_" + str(year) + ".tif") 

    q95_absolute_lr = q95_absolute_lr.isel(band = 0).drop("band").rename({"x":"lon", "y":"lat"})
    q95_absolute_sr = q95_absolute_sr.isel(band = 0).drop("band").rename({"x":"lon", "y":"lat"})
    
    # convert absolute values into relative weights
    q95_relative_lr = q95_absolute_lr/q95_absolute_lr.sum()
    q95_relative_sr = q95_absolute_sr/q95_absolute_sr.sum()

    #
    e_lr_weight = xr.where(q95_mask_lr == 1, ds_e_tmp, np.nan)
    e_sr_weight = xr.where(q95_mask_sr == 1, ds_e_tmp, np.nan)
    
    # calculate the weighted sum of E(source)
    e_lr_weight = e_lr_weight * q95_relative_lr
    e_sr_weight = e_sr_weight * q95_relative_sr

    #
    e_lr_weight = e_lr_weight.sum(dim = ["lat","lon"]).to_pandas()
    e_lr_weight = e_lr_weight.loc[e_lr_weight.index.month.isin([2,3,4,5])]
    e_lr_weight = e_lr_weight.resample("1Y").mean()
    e_lr_weight.index = e_lr_weight.index.year
    e_lr_weight_tmp.append(e_lr_weight)

    e_sr_weight = e_sr_weight.sum(dim = ["lat","lon"]).to_pandas()
    e_sr_weight = e_sr_weight.loc[e_sr_weight.index.month.isin([9,10,11,12])]
    e_sr_weight = e_sr_weight.resample("1Y").mean()
    e_sr_weight.index = e_sr_weight.index.year
    e_sr_weight_tmp.append(e_sr_weight)
    
#%%
e_lr_weight = pd.concat(e_lr_weight_tmp, axis = 0)
e_sr_weight = pd.concat(e_sr_weight_tmp, axis = 0)

#%% 
pre_month = pre_day.resample("1M").mean()
pre_lr = pre_month.loc[pre_month.index.month.isin([3,4,5])]
pre_lr = pre_lr.resample("1Y").mean()
pre_lr.index = pre_lr.index.year

pre_sr = pre_month.loc[pre_month.index.month.isin([10,11,12])]
pre_sr = pre_sr.resample("1Y").mean()
pre_sr.index = pre_sr.index.year

# calculate p/e ratio
percent_pe_lr = (pre_lr/e_lr)*100
percent_pe_sr = (pre_sr/e_sr)*100

#%% create the required data frame
percent_lr_final = pd.concat([pre_lr, e_lr_weight, e_lr], axis = 1)
percent_sr_final = pd.concat([pre_sr, e_sr_weight, e_sr], axis = 1)
percent_lr_final_anm = (percent_lr_final - percent_lr_final.mean())/(percent_lr_final.max() - percent_lr_final.min())
percent_sr_final_anm = (percent_sr_final - percent_sr_final.mean())/(percent_sr_final.max() - percent_sr_final.min())
percent_lr_final_anm_weight = percent_lr_final_anm[[0, 1]]
percent_lr_final_anm_weight.columns = ["P (Sink)", "E (Source)"]
percent_sr_final_anm_weight = percent_sr_final_anm[[0, 1]]
percent_sr_final_anm_weight.columns = ["P (Sink)", "E (Source)"]

#%% plot
cmap = mp.colors.ListedColormap(pl.scientific.diverging.Vik_10.mpl_colors, N = 10)
cmap_2 = mp.colors.ListedColormap(pl.scientific.diverging.Cork_10.mpl_colors, N = 10)

#%% long rains
figure = mp.pyplot.figure(figsize = (9,3.5))
figaxi = figure.add_subplot(1, 1, 1)
percent_lr_final_anm_weight.plot.bar(ax = figaxi, stacked = True,
                           y = ["E (Source)", "P (Sink)"],
                           color = [cmap_2(7), cmap(2)],
                           edgecolor = "black",
                           linewidth = 0.5,
                           clip_on = False)
figaxi.set_xlabel("Year")
figaxi.set_ylabel("Normalized Anomaly (mm)")
mp.pyplot.legend(ncol = 2, edgecolor = "black", loc = "upper right")
mp.pyplot.axhline(y = 0, linewidth = 1.0, color = "black")
mp.pyplot.axvline(x=18, color='black', linestyle='--', linewidth = 0.75)
mp.pyplot.text(x = 24.1, y = 0.37, s="Correlation Coefficient = -0.52")
mp.pyplot.savefig("/Stor1/horn_of_africa/hoa_paper/figures_v3/Figure_7_EvsP_Anomaly_LR_v2.png",
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600)

#%% short rains
figure = mp.pyplot.figure(figsize = (9,3.5))
figaxi = figure.add_subplot(1, 1, 1)
percent_sr_final_anm_weight.plot.bar(ax = figaxi, stacked = True,
                           y = ["E (Source)", "P (Sink)"],
                           color = [cmap_2(7), cmap(2)],
                           edgecolor = "black",
                           linewidth = 0.5,
                           clip_on = False)
figaxi.set_xlabel("Year")
figaxi.set_ylabel("Normalized Anomaly (mm)")
mp.pyplot.legend(ncol = 2, edgecolor = "black", loc = "upper right")
mp.pyplot.axhline(y = 0, linewidth = 1.0, color = "black")
mp.pyplot.axvline(x=21, color='black', linestyle='--', linewidth = 0.75)
mp.pyplot.text(x = 24.1, y = 0.713, s="Correlation Coefficient = -0.15")
mp.pyplot.savefig("/Stor1/horn_of_africa/hoa_paper/figures_v3/Figure_7_EvsP_Anomaly_SR_v2.png",
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600)