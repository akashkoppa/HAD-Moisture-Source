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

#%% read in the hoa masks
shp_hoa = gp.read_file("/Stor1/horn_of_africa/input/study_region_masks/mask_hoa/mask_hoa.shp")

#%% plot the study region (only the drylands)
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
             edgecolor = "black",
             facecolor = "#C19A6B",
             transform = ca.crs.PlateCarree(),
             linewidth = 2,
             zorder = 2)
figure.patch.set_alpha(0)
mp.pyplot.savefig(os.path.join("/Stor1/horn_of_africa/hoa_paper/figures_v2", 
                               "Figure_1_Study_Area.png"), 
                  bbox_inches = "tight",
                  pad_inches = 0.05,
                  dpi = 600,
                  )

