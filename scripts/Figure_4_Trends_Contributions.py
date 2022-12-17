#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Table 1: Recycling Ratio and Contribution Percentages
-------------------------------------------------------------------------------
Author: Akash Koppa
Date: 2022-02-02

"""
#%% import required libraries
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib as mp
import os as os
import palettable as pl
import pingouin as pg

#%% read in the E2P bias corrected data
# horn of africa drylands
e2phoa = xr.open_dataset("/Stor1/horn_of_africa/input/hamster/drylands/monthly/E2P_EPs_1980-2019_allpbl_chirps-gleam_drylands.nc")
e2phoa = e2phoa.drop_dims("bnds")
e2phoa = e2phoa["E2P_EPs"]

#%% read in the mask data
# horn of africa drylands
hoamsk = xr.open_dataset("/Stor1/horn_of_africa/input/study_region_masks/mask_hoa.nc")
hoamsk = hoamsk["mask"]

# oceans 
ocnmsk = xr.open_dataset("/Stor1/horn_of_africa/input/study_region_masks/mask_oceans.nc")
ocnmsk = ocnmsk["mask"]

#%% calculate the recycling ratio for different months
## horn of africa drylands
hoatot = e2phoa.sum(dim = ["lat", "lon"])
hoaslf = xr.where(hoamsk == 1.0, e2phoa, np.nan)
hoaslf = hoaslf.sum(dim = ["lat", "lon"])
hoarer = pd.Series(np.array(hoaslf), index = np.array(hoaslf.time)) / \
         pd.Series(np.array(hoatot), index = np.array(hoaslf.time))
hoarer = hoarer * 100
# MAM
mamhoa = hoarer.loc[(hoarer.index.month >= 3) & (hoarer.index.month <= 5)]
mamhoa = mamhoa.groupby(mamhoa.index.year).mean()
# OND
ondhoa = hoarer.loc[(hoarer.index.month >= 10) & (hoarer.index.month <= 12)]
ondhoa = ondhoa.groupby(ondhoa.index.year).mean()

# March 
marhoa = hoarer.loc[(hoarer.index.month == 3)]
#marhoa = marhoa.mean()

# April
aprhoa = hoarer.loc[(hoarer.index.month == 4)]
#aprhoa = aprhoa.mean()

# May
mayhoa = hoarer.loc[(hoarer.index.month == 5)]
#mayhoa = mayhoa.mean()

# October
octhoa = hoarer.loc[(hoarer.index.month == 10)]
#octhoa = octhoa.mean()

# November
novhoa = hoarer.loc[(hoarer.index.month == 11)]
#novhoa = novhoa.mean()

# December
dechoa = hoarer.loc[(hoarer.index.month == 12)]
#dechoa = dechoa.mean()

#%% Ocean Contribution
## horn of africa drylands
hoaocn = xr.where(ocnmsk == 1.0, e2phoa, np.nan)
hoaocn = hoaocn.sum(dim = ["lat", "lon"])
hoaocn = pd.Series(np.array(hoaocn), index = np.array(hoaocn.time)) / \
         pd.Series(np.array(hoatot), index = np.array(hoaocn.time))
hoaocn = hoaocn * 100
# MAM
mamhoa_ocn = hoaocn.loc[(hoaocn.index.month >= 3) & (hoaocn.index.month <= 5)]
mamhoa_ocn = mamhoa_ocn.groupby(mamhoa_ocn.index.year).mean()

# OND
ondhoa_ocn = hoaocn.loc[(hoaocn.index.month >= 10) & (hoaocn.index.month <= 12)]
ondhoa_ocn = ondhoa_ocn.groupby(ondhoa_ocn.index.year).mean()

# March 
marhoa_ocn = hoaocn.loc[(hoaocn.index.month == 3)]
#marhoa_ocn = marhoa_ocn.mean()

# April 
aprhoa_ocn = hoaocn.loc[(hoaocn.index.month == 4)]
#aprhoa_ocn = aprhoa_ocn.mean()

# May
mayhoa_ocn = hoaocn.loc[(hoaocn.index.month == 5)]
#mayhoa_ocn = mayhoa_ocn.mean()

# October
octhoa_ocn = hoaocn.loc[(hoaocn.index.month == 10)]
#octhoa_ocn = octhoa_ocn.mean()

# November
novhoa_ocn = hoaocn.loc[(hoaocn.index.month == 11)]
#novhoa_ocn = novhoa_ocn.mean()

# December
dechoa_ocn = hoaocn.loc[(hoaocn.index.month == 12)]
#dechoa_ocn = dechoa_ocn.mean()

#%% MAM 
mamall = pd.concat([mamhoa, 100-(mamhoa + mamhoa_ocn), mamhoa_ocn], axis = 1)
mamall.columns = ["Local Recycling", "External Land", "Ocean"]

# March 
marall = pd.concat([marhoa, 100-(marhoa + marhoa_ocn), marhoa_ocn],  axis = 1)
marall.columns = ["Local Recycling", "External Land", "Ocean"]
marall.index = mamall.index

# April 
aprall = pd.concat([aprhoa, 100-(aprhoa + aprhoa_ocn), aprhoa_ocn], axis = 1)
aprall.columns = ["Local Recycling", "External Land", "Ocean"]
aprall.index = mamall.index

# May
mayall = pd.concat([mayhoa, 100-(mayhoa + mayhoa_ocn), mayhoa_ocn], axis = 1)
mayall.columns = ["Local Recycling", "External Land", "Ocean"]
mayall.index = mamall.index

#%% OND
ondall = pd.concat([ondhoa, 100-(ondhoa + ondhoa_ocn), ondhoa_ocn], axis = 1)
ondall.columns = ["Local Recycling", "External Land", "Ocean"]

# October
octall = pd.concat([octhoa, 100-(octhoa + octhoa_ocn), octhoa_ocn], axis = 1)
octall.columns = ["Local Recycling", "External Land", "Ocean"]
octall.index = ondall.index

# November
novall = pd.concat([novhoa, 100-(novhoa + novhoa_ocn), novhoa_ocn], axis = 1)
novall.columns = ["Local Recycling", "External Land", "Ocean"]
novall.index = ondall.index

# December
decall = pd.concat([dechoa, 100-(dechoa + dechoa_ocn), dechoa_ocn], axis = 1)
decall.columns = ["Local Recycling", "External Land", "Ocean"]
decall.index = ondall.index

#%% read in the flexpart data 
flxfil = "/Stor1/horn_of_africa/input/hamster/drylands/daily"
ds = xr.open_mfdataset(os.path.join(flxfil, "*.nc"), chunks = {"lat": 50, "lon": 50, "time" :-1})

#%% read in the mask
maskfi = "/Stor1/horn_of_africa/input/study_region_masks/mask_hoa.nc"
maskda = xr.open_dataset(maskfi)
maskda = maskda["mask"]
#maskda = maskda.rio.set_crs(4326)

#%% calculate precipitation totals 
# long rain
pre_month = ds["E2P_EPs"].resample(time = "1M").sum()
pre_lr = pre_month.sel(time = pre_month.time.dt.month.isin([3,4,5]))
#pre_clim = ds["E2P_EPs"].groupby("time.month").mean()
pre_lr.compute()
pre_lr_annual = pre_lr.groupby("time.year").sum()
pre_lr_annual = pre_lr_annual.sum(dim = ["lat","lon"]).to_pandas()
pre_lr_annual = pre_lr_annual/np.array(maskda.where(maskda == 1).count())

#%% short rain
pre_sr = pre_month.sel(time = pre_month.time.dt.month.isin([10,11,12]))
#pre_clim = ds["E2P_EPs"].groupby("time.month").mean()
pre_sr.compute()
pre_sr_annual = pre_sr.groupby("time.year").sum()
pre_sr_annual = pre_sr_annual.sum(dim = ["lat","lon"]).to_pandas()
pre_sr_annual = pre_sr_annual/np.array(maskda.where(maskda == 1).count())

#%% calculate precipitation totals for every month
# long rain
pre_month = ds["E2P_EPs"].resample(time = "1M").sum()
# March
pre_mar = pre_month.sel(time = pre_month.time.dt.month.isin([3]))
pre_mar = pre_mar.sum(dim = ["lat","lon"]).to_pandas()
pre_mar = pre_mar/np.array(maskda.where(maskda == 1).count())
pre_mar.index = marall.index

# April
pre_apr = pre_month.sel(time = pre_month.time.dt.month.isin([4]))
pre_apr = pre_apr.sum(dim = ["lat","lon"]).to_pandas()
pre_apr = pre_apr/np.array(maskda.where(maskda == 1).count())
pre_apr.index = marall.index

# May
pre_may = pre_month.sel(time = pre_month.time.dt.month.isin([5]))
pre_may = pre_may.sum(dim = ["lat","lon"]).to_pandas()
pre_may = pre_may/np.array(maskda.where(maskda == 1).count())
pre_may.index = marall.index

#%% short rain
# October
pre_oct = pre_month.sel(time = pre_month.time.dt.month.isin([10]))
pre_oct = pre_oct.sum(dim = ["lat","lon"]).to_pandas()
pre_oct = pre_oct/np.array(maskda.where(maskda == 1).count())
pre_oct.index = marall.index

# November
pre_nov = pre_month.sel(time = pre_month.time.dt.month.isin([11]))
pre_nov = pre_nov.sum(dim = ["lat","lon"]).to_pandas()
pre_nov = pre_nov/np.array(maskda.where(maskda == 1).count())
pre_nov.index = marall.index

# December
pre_dec = pre_month.sel(time = pre_month.time.dt.month.isin([12]))
pre_dec = pre_dec.sum(dim = ["lat","lon"]).to_pandas()
pre_dec = pre_dec/np.array(maskda.where(maskda == 1).count())
pre_dec.index = marall.index

#%% calculate the absolute values of contributions
# long rain 
mam_con = pd.concat([pre_lr_annual * (mamall[i]/100) for i in mamall.keys()], axis = 1)
mam_con.columns = mamall.keys()
# short rain 
ond_con = pd.concat([pre_sr_annual * (ondall[i]/100) for i in ondall.keys()], axis = 1)
ond_con.columns = ondall.keys()

# March
mar_con = pd.concat([pre_mar * (marall[i]/100) for i in marall.keys()], axis =1)
mar_con.columns = marall.keys()

# April
apr_con = pd.concat([pre_apr * (aprall[i]/100) for i in aprall.keys()], axis =1)
apr_con.columns = aprall.keys()

# May
may_con = pd.concat([pre_may * (marall[i]/100) for i in mayall.keys()], axis =1)
may_con.columns = mayall.keys()

# October
oct_con = pd.concat([pre_mar * (octall[i]/100) for i in octall.keys()], axis =1)
oct_con.columns = octall.keys()

# November
nov_con = pd.concat([pre_nov * (novall[i]/100) for i in novall.keys()], axis =1)
nov_con.columns = novall.keys()

# December
dec_con = pd.concat([pre_dec * (decall[i]/100) for i in decall.keys()], axis =1)
dec_con.columns = decall.keys()

#%% plot the figure
# get required number of colors
#cmap = mp.colors.ListedColormap(pl.scientific.sequential.GrayC_10_r.mpl_colors, N = 10)
#cmap_1 = mp.colors.ListedColormap(pl.scientific.diverging.Vik_10.mpl_colors, N = 10)
cmap = mp.colors.ListedColormap(pl.scientific.sequential.Bilbao_10.mpl_colors, N = 10)
cmap_1 = mp.colors.ListedColormap(pl.scientific.sequential.Oslo_10.mpl_colors, N = 10)

#%% plot the contributions
# Long Rains
figure = mp.pyplot.figure(figsize = (9,3.5))
figaxi = figure.add_subplot(1, 1, 1)
#pre_lr.plot(ax = figaxi, color = [cmap(1), cmap(2), cmap(3)], 
#            style = ['^-','<-', 'v-'])
mam_con.plot.bar(ax = figaxi, stacked = True, 
                color = [cmap(4), cmap(5), cmap_1(6)],
                edgecolor = "black",
                linewidth = 0.3)
#pre_mar.plot(ax =figaxi, color = [cmap_1(2)], 
#                   style = ['o-'],
#                   secondary_y = False, 
#                   use_index = False, 
#                   mark_right = False)
figaxi.set_xlabel("Month")
figaxi.set_ylabel("Precipitation (mm)")
mp.pyplot.legend(ncol = 2, edgecolor = "black", loc = "upper right")
figaxi.set_xticklabels(mam_con.index, rotation = 90)
mp.pyplot.axvline(x=18, color='black', linestyle='--', linewidth = 0.75)
#figaxi.legend().get_frame().set_edgecolor("black")
#figaxi.right_ax.set_ylabel("Max Temperature (deg C)")
mp.pyplot.savefig("/Stor1/horn_of_africa/hoa_paper/figures_v2/Figure_4_Time_Series_LR.png",
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600)

#%% Short Rains
figure = mp.pyplot.figure(figsize = (9,3.5))
figaxi = figure.add_subplot(1, 1, 1)
#pre_lr.plot(ax = figaxi, color = [cmap(1), cmap(2), cmap(3)], 
#            style = ['^-','<-', 'v-'])
ond_con.plot.bar(ax = figaxi, stacked = True, 
                color = [cmap(4), cmap(5), cmap_1(6)],
                edgecolor = "black",
                linewidth = 0.3)
#pre_mar.plot(ax =figaxi, color = [cmap_1(2)], 
#                   style = ['o-'],
#                   secondary_y = False, 
#                   use_index = False, 
#                   mark_right = False)
figaxi.set_xlabel("Month")
figaxi.set_ylabel("Precipitation (mm)")
mp.pyplot.legend(ncol = 2, edgecolor = "black", loc = "upper right")
figaxi.set_xticklabels(mam_con.index, rotation = 90)
mp.pyplot.axvline(x=21, color='black', linestyle='--', linewidth = 0.75)
#figaxi.legend().get_frame().set_edgecolor("black")
#figaxi.right_ax.set_ylabel("Max Temperature (deg C)")
mp.pyplot.savefig("/Stor1/horn_of_africa/hoa_paper/figures_v2/Figure_4_Time_Series_SR.png",
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600)

"""
#%% calculate the trends in contributions
#%% long rains
mam_reg_local = pg.linear_regression(np.array(mam_con["Local Recycling"].values.reshape(-1, 1)).flatten(),
                                     np.array(mam_con.index.values.reshape(-1,1)).flatten())
mam_reg_ext = pg.linear_regression(np.array(mam_con["External Land"].values.reshape(-1, 1)).flatten(),
                                     np.array(mam_con.index.values.reshape(-1,1)).flatten())
mam_reg_ocn = pg.linear_regression(np.array(mam_con["Ocean"].values.reshape(-1, 1)).flatten(),
                                     np.array(mam_con.index.values.reshape(-1,1)).flatten())

#%% short rains
ond_con = ond_con.drop(1997)
ondall = ondall.drop(1997)
#%%
ond_reg_local = pg.linear_regression(np.array(ond_con["Local Recycling"].values.reshape(-1, 1)).flatten(),
                                     np.array(ond_con.index.values.reshape(-1,1)).flatten())
ond_reg_ext = pg.linear_regression(np.array(ond_con["External Land"].values.reshape(-1, 1)).flatten(),
                                     np.array(ond_con.index.values.reshape(-1,1)).flatten())
ond_reg_ocn = pg.linear_regression(np.array(ond_con["Ocean"].values.reshape(-1, 1)).flatten(),
                                     np.array(ond_con.index.values.reshape(-1,1)).flatten())
#%% Draw two horizontal bar plots showing the percentage contribution
"""
"""
#%% plot the contributions
# March
figure = mp.pyplot.figure(figsize = (9,3.5))
figaxi = figure.add_subplot(1, 1, 1)
#pre_lr.plot(ax = figaxi, color = [cmap(1), cmap(2), cmap(3)], 
#            style = ['^-','<-', 'v-'])
mar_con.plot.bar(ax = figaxi, stacked = True, 
                color = [cmap(1), cmap(4), cmap(7)])
#pre_mar.plot(ax =figaxi, color = [cmap_1(2)], 
#                   style = ['o-'],
#                   secondary_y = False, 
#                   use_index = False, 
#                   mark_right = False)
figaxi.set_xlabel("Month")
figaxi.set_ylabel("Precipitation (mm/month)")
mp.pyplot.legend(ncol = 2, edgecolor = "black", loc = "upper right")
figaxi.set_xticklabels(mar_con.index, rotation = 90)
#figaxi.legend().get_frame().set_edgecolor("black")
#figaxi.right_ax.set_ylabel("Max Temperature (deg C)")
mp.pyplot.savefig("/Stor1/horn_of_africa/hoa_paper/figures/Figure_4_Time_Series_Mar.png",
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600)

# April
figure = mp.pyplot.figure(figsize = (9,3.5))
figaxi = figure.add_subplot(1, 1, 1)
#pre_lr.plot(ax = figaxi, color = [cmap(1), cmap(2), cmap(3)], 
#            style = ['^-','<-', 'v-'])
apr_con.plot.bar(ax = figaxi, stacked = True, 
                color = [cmap(1), cmap(4), cmap(7)])
#pre_mar.plot(ax =figaxi, color = [cmap_1(2)], 
#                   style = ['o-'],
#                   secondary_y = False, 
#                   use_index = False, 
#                   mark_right = False)
figaxi.set_xlabel("Month")
figaxi.set_ylabel("Precipitation (mm/month)")
mp.pyplot.legend(ncol = 2, edgecolor = "black", loc = "upper right")
figaxi.set_xticklabels(apr_con.index, rotation = 90)
#figaxi.legend().get_frame().set_edgecolor("black")
#figaxi.right_ax.set_ylabel("Max Temperature (deg C)")
mp.pyplot.savefig("/Stor1/horn_of_africa/hoa_paper/figures/Figure_4_Time_Series_Apr.png",
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600)

# May
figure = mp.pyplot.figure(figsize = (9,3.5))
figaxi = figure.add_subplot(1, 1, 1)
#pre_lr.plot(ax = figaxi, color = [cmap(1), cmap(2), cmap(3)], 
#            style = ['^-','<-', 'v-'])
may_con.plot.bar(ax = figaxi, stacked = True, 
                color = [cmap(1), cmap(4), cmap(7)])
#pre_mar.plot(ax =figaxi, color = [cmap_1(2)], 
#                   style = ['o-'],
#                   secondary_y = False, 
#                   use_index = False, 
#                   mark_right = False)
figaxi.set_xlabel("Month")
figaxi.set_ylabel("Precipitation (mm/month)")
mp.pyplot.legend(ncol = 2, edgecolor = "black", loc = "upper right")
figaxi.set_xticklabels(may_con.index, rotation = 90)
#figaxi.legend().get_frame().set_edgecolor("black")
#figaxi.right_ax.set_ylabel("Max Temperature (deg C)")
mp.pyplot.savefig("/Stor1/horn_of_africa/hoa_paper/figures/Figure_4_Time_Series_May.png",
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600)

# October
figure = mp.pyplot.figure(figsize = (9,3.5))
figaxi = figure.add_subplot(1, 1, 1)
#pre_lr.plot(ax = figaxi, color = [cmap(1), cmap(2), cmap(3)], 
#            style = ['^-','<-', 'v-'])
oct_con.plot.bar(ax = figaxi, stacked = True, 
                color = [cmap(1), cmap(4), cmap(7)])
#pre_mar.plot(ax =figaxi, color = [cmap_1(2)], 
#                   style = ['o-'],
#                   secondary_y = False, 
#                   use_index = False, 
#                   mark_right = False)
figaxi.set_xlabel("Month")
figaxi.set_ylabel("Precipitation (mm/month)")
mp.pyplot.legend(ncol = 2, edgecolor = "black", loc = "upper right")
figaxi.set_xticklabels(oct_con.index, rotation = 90)
#figaxi.legend().get_frame().set_edgecolor("black")
#figaxi.right_ax.set_ylabel("Max Temperature (deg C)")
mp.pyplot.savefig("/Stor1/horn_of_africa/hoa_paper/figures/Figure_4_Time_Series_Oct.png",
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600)

# November
figure = mp.pyplot.figure(figsize = (9,3.5))
figaxi = figure.add_subplot(1, 1, 1)
#pre_lr.plot(ax = figaxi, color = [cmap(1), cmap(2), cmap(3)], 
#            style = ['^-','<-', 'v-'])
nov_con.plot.bar(ax = figaxi, stacked = True, 
                color = [cmap(1), cmap(4), cmap(7)])
#pre_mar.plot(ax =figaxi, color = [cmap_1(2)], 
#                   style = ['o-'],
#                   secondary_y = False, 
#                   use_index = False, 
#                   mark_right = False)
figaxi.set_xlabel("Month")
figaxi.set_ylabel("Precipitation (mm/month)")
mp.pyplot.legend(ncol = 2, edgecolor = "black", loc = "upper right")
figaxi.set_xticklabels(nov_con.index, rotation = 90)
#figaxi.legend().get_frame().set_edgecolor("black")
#figaxi.right_ax.set_ylabel("Max Temperature (deg C)")
mp.pyplot.savefig("/Stor1/horn_of_africa/hoa_paper/figures/Figure_4_Time_Series_Nov.png",
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600)

# December
figure = mp.pyplot.figure(figsize = (9,3.5))
figaxi = figure.add_subplot(1, 1, 1)
#pre_lr.plot(ax = figaxi, color = [cmap(1), cmap(2), cmap(3)], 
#            style = ['^-','<-', 'v-'])
dec_con.plot.bar(ax = figaxi, stacked = True, 
                color = [cmap(1), cmap(4), cmap(7)])
#pre_mar.plot(ax =figaxi, color = [cmap_1(2)], 
#                   style = ['o-'],
#                   secondary_y = False, 
#                   use_index = False, 
#                   mark_right = False)
figaxi.set_xlabel("Month")
figaxi.set_ylabel("Precipitation (mm/month)")
mp.pyplot.legend(ncol = 2, edgecolor = "black", loc = "upper right")
figaxi.set_xticklabels(dec_con.index, rotation = 90)
#figaxi.legend().get_frame().set_edgecolor("black")
#figaxi.right_ax.set_ylabel("Max Temperature (deg C)")
mp.pyplot.savefig("/Stor1/horn_of_africa/hoa_paper/figures/Figure_4_Time_Series_Dec.png",
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600)

"""












