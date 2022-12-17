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
import pyhomogeneity as hg
import sklearn.linear_model as lr
import statsmodels.api as sm
import pingouin as pg

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

#%% calculate precipitation totals 
# long rain
pre_month = ds["E2P_EPs"].resample(time = "1M").sum()
# March
pre_mar = pre_month.sel(time = pre_month.time.dt.month.isin([3]))
pre_mar = pre_mar.groupby("time.year").sum(dim = ["lat", "lon"]).to_pandas()
pre_mar = pre_mar/np.array(maskda.where(maskda == 1).count())
pre_mar.index = pre_mar.index.year
pre_mar_anm = pre_mar - pre_mar.mean()
pre_mar_anm.name = "March"
# April
pre_apr = pre_month.sel(time = pre_month.time.dt.month.isin([4]))
pre_apr = pre_apr.groupby("time.year").sum(dim = ["lat", "lon"]).to_pandas()
pre_apr = pre_apr/np.array(maskda.where(maskda == 1).count())
pre_apr.index = pre_apr.index.year
pre_apr_anm = pre_apr - pre_apr.mean()
pre_apr_anm.name = "April"
# May
pre_may = pre_month.sel(time = pre_month.time.dt.month.isin([5]))
pre_may = pre_may.groupby("time.year").sum(dim = ["lat", "lon"]).to_pandas()
pre_may = pre_may/np.array(maskda.where(maskda == 1).count())
pre_may.index = pre_may.index.year
pre_may_anm = pre_may - pre_may.mean()
pre_may_anm.name = "May"

#%% long rain
pre_lr = pre_month.sel(time = pre_month.time.dt.month.isin([3,4,5]))
pre_lr.compute()
pre_lr_annual = pre_lr.groupby("time.year").sum()
pre_lr_annual = pre_lr_annual.sum(dim = ["lat","lon"]).to_pandas()
pre_lr_annual = pre_lr_annual/np.array(maskda.where(maskda == 1).count())
pre_lr_annual = pre_lr_annual - pre_lr_annual.mean()
pre_lr_annual.name = "Long Rains"

#%% short rain
# October
pre_oct = pre_month.sel(time = pre_month.time.dt.month.isin([10]))
pre_oct = pre_oct.groupby("time.year").sum(dim = ["lat", "lon"]).to_pandas()
pre_oct = pre_oct/np.array(maskda.where(maskda == 1).count())
pre_oct.index = pre_oct.index.year
pre_oct_anm = pre_oct - pre_oct.mean()
pre_oct_anm.name = "October"
# November
pre_nov = pre_month.sel(time = pre_month.time.dt.month.isin([11]))
pre_nov = pre_nov.groupby("time.year").sum(dim = ["lat", "lon"]).to_pandas()
pre_nov = pre_nov/np.array(maskda.where(maskda == 1).count())
pre_nov.index = pre_nov.index.year
pre_nov_anm = pre_nov - pre_nov.mean()
pre_nov_anm.name = "November"
# December
pre_dec = pre_month.sel(time = pre_month.time.dt.month.isin([12]))
pre_dec = pre_dec.groupby("time.year").sum(dim = ["lat", "lon"]).to_pandas()
pre_dec = pre_dec/np.array(maskda.where(maskda == 1).count())
pre_dec.index = pre_dec.index.year
pre_dec_anm = pre_dec - pre_dec.mean()
pre_dec_anm.name = "December"

#%% short rain
pre_sr = pre_month.sel(time = pre_month.time.dt.month.isin([10,11,12]))
pre_sr.compute()
pre_sr_annual = pre_sr.groupby("time.year").sum()
pre_sr_annual = pre_sr_annual.sum(dim = ["lat","lon"]).to_pandas()
pre_sr_annual = pre_sr_annual/np.array(maskda.where(maskda == 1).count())
pre_sr_annual = pre_sr_annual - pre_sr_annual.mean()
pre_sr_annual.name = "Short Rains"
# remove 1997 from the short rains
del pre_sr_annual[1997]
#pre_sr_annual[1997] = np.nan

#%% identify the change point using Pettitt's test
pt_lr = hg.pettitt_test(pre_lr_annual, alpha = 0.05)
pt_sr = hg.pettitt_test(pre_sr_annual, alpha = 0.05)

#%% calculate and plot the trend line before and after the change point
# long rains
regressor = lr.LinearRegression()
regressor1 = lr.LinearRegression()
regressor2 = lr.LinearRegression()
regressor3 = lr.LinearRegression()
lr_bcp_reg = regressor.fit(pre_lr_annual.index[0:(pt_lr.cp)].values.reshape(-1, 1),
                          pre_lr_annual.iloc[0:(pt_lr.cp)].values.reshape(-1, 1))
lr_bcp_reg_pred = lr_bcp_reg.predict(pre_lr_annual.index[0:(pt_lr.cp)].values.reshape(-1, 1))

lr_acp_reg = regressor1.fit(pre_lr_annual.index[(pt_lr.cp):(pre_lr_annual.size)].values.reshape(-1, 1),
                          pre_lr_annual.iloc[(pt_lr.cp):(pre_lr_annual.size)].values.reshape(-1, 1))
lr_acp_reg_pred = lr_acp_reg.predict(pre_lr_annual.index[(pt_lr.cp):(pre_lr_annual.size)].values.reshape(-1, 1))

# short rains
sr_bcp_reg = regressor2.fit(pre_sr_annual.index[0:(pt_sr.cp)].values.reshape(-1, 1),
                          pre_sr_annual.iloc[0:(pt_sr.cp)].values.reshape(-1, 1))
#sr_bcp_reg_pred = sr_bcp_reg.predict(pre_sr_annual.index[0:(pt_sr.cp)].values.reshape(-1, 1))
sr_bcp_reg_pred = sr_bcp_reg.predict(pre_lr_annual.index[0:(pt_sr.cp+1)].values.reshape(-1, 1))

sr_acp_reg = regressor3.fit(pre_sr_annual.index[(pt_sr.cp):(pre_sr_annual.size)].values.reshape(-1, 1),
                          pre_sr_annual.iloc[(pt_sr.cp):(pre_sr_annual.size)].values.reshape(-1, 1))
sr_acp_reg_pred = sr_acp_reg.predict(pre_sr_annual.index[(pt_sr.cp):(pre_sr_annual.size)].values.reshape(-1, 1))

#%% using stats models for p-values
#lr_bcp_reg_statsmodel = sm.OLS(pre_lr_annual.iloc[0:(pt_lr.cp)].values.reshape(-1, 1),
#                               pre_lr_annual.index[0:(pt_lr.cp)].values.reshape(-1, 1))
#lr_bcp_fit_statsmodel = lr_bcp_reg_statsmodel.fit()
#lr_bcp_reg_pred = lr_bcp_fit_statsmodel.predict(pre_lr_annual.index[0:(pt_lr.cp)].values.reshape(-1, 1))

#lr_acp_reg_statsmodel = sm.OLS(pre_lr_annual.iloc[(pt_lr.cp):(pre_lr_annual.size)].values.reshape(-1, 1),
#                               pre_lr_annual.index[(pt_lr.cp):(pre_lr_annual.size)].values.reshape(-1, 1))
#lr_acp_fit_statsmodel = lr_acp_reg_statsmodel.fit()
#lr_acp_reg_pred = lr_acp_fit_statsmodel.predict(pre_lr_annual.index[(pt_lr.cp):(pre_lr_annual.size)].values.reshape(-1, 1))

#sr_bcp_reg_statsmodel = sm.OLS(pre_sr_annual.iloc[0:(pt_sr.cp)].values.reshape(-1, 1),
#                               pre_sr_annual.index[0:(pt_sr.cp)].values.reshape(-1, 1))
#sr_bcp_fit_statsmodel = sr_bcp_reg_statsmodel.fit()
#sr_bcp_reg_pred = sr_bcp_fit_statsmodel.predict(pre_lr_annual.index[0:(pt_sr.cp+1)].values.reshape(-1, 1))

#sr_acp_reg_statsmodel = sm.OLS(pre_sr_annual.iloc[(pt_sr.cp):(pre_sr_annual.size)].values.reshape(-1, 1),
#                               pre_sr_annual.index[(pt_sr.cp):(pre_sr_annual.size)].values.reshape(-1, 1))
#sr_acp_fit_statsmodel = sr_acp_reg_statsmodel.fit()
#sr_acp_reg_pred = sr_acp_fit_statsmodel.predict(pre_sr_annual.index[(pt_sr.cp):(pre_sr_annual.size)].values.reshape(-1, 1))

#%% pingouin
lr_bcp_reg_pingouin = pg.linear_regression(np.array(pre_lr_annual.index[0:(pt_lr.cp)].values.reshape(-1, 1)).flatten(),
                                           np.array(pre_lr_annual.iloc[0:(pt_lr.cp)].values.reshape(-1, 1)).flatten())

lr_acp_reg_pingouin = pg.linear_regression(np.array(pre_lr_annual.index[(pt_lr.cp):(pre_lr_annual.size)].values.reshape(-1, 1)).flatten(),
                                           np.array(pre_lr_annual.iloc[(pt_lr.cp):(pre_lr_annual.size)].values.reshape(-1, 1)).flatten())

sr_bcp_reg_pingouin = pg.linear_regression(np.array(pre_sr_annual.index[0:(pt_sr.cp)].values.reshape(-1, 1)).flatten(),
                                           np.array(pre_sr_annual.iloc[0:(pt_sr.cp)].values.reshape(-1, 1)).flatten())

sr_acp_reg_pingouin = pg.linear_regression(np.array(pre_sr_annual.index[(pt_sr.cp):(pre_sr_annual.size)].values.reshape(-1, 1)).flatten(),
                                           np.array(pre_sr_annual.iloc[(pt_sr.cp):(pre_sr_annual.size)].values.reshape(-1, 1)).flatten())

#%% create pandas series for the predicted values
lr_reg_pred_all = pd.Series(np.append(lr_bcp_reg_pred, lr_acp_reg_pred), index = pre_lr_annual.index)
lr_reg_pred_all.iloc[pt_lr.cp-1] = np.nan
lr_reg_pred_all.name = "Mean"
#sr_reg_pred_all = pd.Series(np.append(sr_bcp_reg_pred, sr_acp_reg_pred), index = pre_sr_annual.index)
sr_reg_pred_all = pd.Series(np.append(sr_bcp_reg_pred, sr_acp_reg_pred), index = pre_lr_annual.index)
sr_reg_pred_all.iloc[pt_sr.cp] = np.nan
sr_reg_pred_all.name = "Mean"

lr_reg_pred_all.iloc[0:pt_lr.cp] = pre_lr_annual.iloc[0:pt_lr.cp].mean()
lr_reg_pred_all.iloc[pt_lr.cp:lr_reg_pred_all.size] = pre_lr_annual.iloc[pt_lr.cp:lr_reg_pred_all.size].mean()
lr_reg_pred_all.iloc[pt_lr.cp-1] = np.nan

sr_reg_pred_all.iloc[0:pt_sr.cp] = pre_sr_annual.iloc[0:pt_sr.cp].mean()
sr_reg_pred_all.iloc[pt_sr.cp:sr_reg_pred_all.size] = pre_sr_annual.iloc[pt_sr.cp:sr_reg_pred_all.size].mean()
sr_reg_pred_all.iloc[pt_sr.cp] = np.nan

#%% get required number of colors
cmap = mp.colors.ListedColormap(pl.scientific.sequential.GrayC_10_r.mpl_colors, N = 10)
cmap_1 = mp.colors.ListedColormap(pl.scientific.diverging.Cork_10.mpl_colors, N = 10)
cmap_2 = mp.colors.ListedColormap(pl.scientific.diverging.Vik_10.mpl_colors, N = 10)

#%% plot the anomalies
# long rains
pre_lr = pd.concat([pre_mar_anm, pre_apr_anm, pre_may_anm], axis = 1)
figure = mp.pyplot.figure(figsize = (9,4.5))
figaxi = figure.add_subplot(1, 1, 1)
#pre_lr.plot(ax = figaxi, color = [cmap(1), cmap(2), cmap(3)], 
#            style = ['^-','<-', 'v-'])
#pre_lr.rolling(5).mean().loc[range(1987, 2016)].plot.bar(ax = figaxi, stacked = True, 
#                color = [cmap(1), cmap(4), cmap(7)])
#pre_lr_annual.rolling(5).mean().loc[range(1987, 2016)].plot(ax =figaxi, color = [cmap_1(2)], 
#                   style = ['o-'],
#                   secondary_y = False, 
#                   use_index = False, 
#                   mark_right = False)
pre_lr.plot.bar(ax = figaxi, stacked = True, 
                color = [cmap(4), cmap(6), cmap(8)],
                edgecolor = "black",
                linewidth = 0.3)
pre_lr_annual.plot(ax =figaxi, color = [cmap_2(1)], 
                   style = ['x-'],
                   secondary_y = False, 
                   use_index = False, 
                   mark_right = False)
lr_reg_pred_all.plot(ax =figaxi, color = 'red', 
                   style = ['-'],
                   secondary_y = False, 
                   use_index = False, 
                   mark_right = False,
                   label = '_nolegend_')
mp.pyplot.axhline(y=0.0, color='black', linestyle='-', linewidth = 0.75)
mp.pyplot.axvline(x=18, color='black', linestyle='--', linewidth = 0.75)
figaxi.set_xlabel("Year")
figaxi.set_ylabel("Precipitation Anomaly (mm)")
#figaxi.text(5.15,-88, r'y = 0.22x - 423.57', fontsize = 10)
#figaxi.text(23.75,-88, r'y = 1.80x - 3649.73', fontsize = 10)
mp.pyplot.legend(ncol = 2, edgecolor = "black", loc = "upper right")
figaxi.set_xticklabels(pre_lr.loc[range(1980, 2017)].index, rotation = 90)
#figaxi.legend().get_frame().set_edgecolor("black")
#figaxi.right_ax.set_ylabel("Max Temperature (deg C)")
mp.pyplot.savefig("/Stor1/horn_of_africa/hoa_paper/figures_v3/Figure_2_Anomaly_LR.png",
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600)

#%% short rains
pre_sr = pd.concat([pre_oct_anm, pre_nov_anm, pre_dec_anm], axis = 1)
figure = mp.pyplot.figure(figsize = (9,4.5))
figaxi = figure.add_subplot(1, 1, 1)
#pre_lr.plot(ax = figaxi, color = [cmap(1), cmap(2), cmap(3)], 
#            style = ['^-','<-', 'v-'])
#pre_sr.rolling(5).mean().loc[range(1987, 2016)].plot.bar(ax = figaxi, stacked = True, 
#                color = [cmap(1), cmap(4), cmap(7)])
#pre_sr_annual.rolling(5).mean().loc[range(1987, 2016)].plot(ax =figaxi, color = [cmap_1(7)], 
#                   style = ['x-'],
#                   secondary_y = False, 
#                   use_index = False, 
#                   mark_right = False)
pre_sr.plot.bar(ax = figaxi, stacked = True, 
                color = [cmap(4), cmap(6), cmap(8)],
                edgecolor = "black",
                linewidth = 0.3)
pre_sr_annual.plot(ax =figaxi, color = [cmap_1(8)], 
                   style = ['x-'],
                   secondary_y = False, 
                   use_index = False, 
                   mark_right = False)
sr_reg_pred_all.plot(ax =figaxi, color = 'red', 
                   style = ['-'],
                   secondary_y = False, 
                   use_index = False, 
                   mark_right = False,
                   label = '_nolegend_')
mp.pyplot.axhline(y=0.0, color='black', linestyle='-', linewidth = 0.75)
mp.pyplot.axvline(x=21, color='black', linestyle='--', linewidth = 0.75)
figaxi.set_xlabel("Year")
figaxi.set_ylabel("Precipitation Anomaly (mm)")
figaxi.text(5.15,-88, r'y = 1.77x - 3550.91', fontsize = 10)
figaxi.text(21.05,-88, r'y = -0.62x - 1278.03', fontsize = 10)
mp.pyplot.legend(ncol = 2, edgecolor = "black", loc = "upper right")
figaxi.set_xticklabels(pre_lr.rolling(5).mean().loc[range(1980, 2017)].index, rotation = 90)
#figaxi.legend().get_frame().set_edgecolor("black")
#figaxi.right_ax.set_ylabel("Max Temperature (deg C)")
mp.pyplot.savefig("/Stor1/horn_of_africa/hoa_paper/figures_v3/Figure_2_Anomaly_SR.png",
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600)




