#!/usr/bin/env python
# coding: utf-8

## This script does infilling for the 
#- LGMR ensemble members, annually mean
#- so it loops through ens members instead of months.
#- The purpose is to use annual mean infilled SST with green funcs.
#- I will potentially need to adjust this if i want
#- to infill ensemble members monthly for model runs.


#########
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
import xesmf as xe
import cartopy.crs as ccrs
import copy
import pandas as pd
import datetime

import pykrige.ok as pyok
from sklearn.metrics.pairwise import haversine_distances

import datetime
begin_time = datetime.datetime.now()


## ensemble member version
## this will determine name of what is saved

path = '/home/disk/sipn/vcooper/nobackup/lgm/LGMR/'
fname = 'LGMR_SST_holo_0-4ka_members.nc'
ds_holo_sst_all = xr.open_dataset(path + fname).sst_ann

fname = 'LGMR_SST_lgm_19-23ka_members.nc'
ds_lgm_sst_all = xr.open_dataset(path + fname).sst_ann


## BEGIN FOR LOOP
for iens in range(len(ds_holo_sst_all.nEns.values)):
    #iens=0
    memstring = str(iens+1).zfill(2)

    ds_holo_sst = ds_holo_sst_all.isel(nEns=iens)
    ds_lgm_sst = ds_lgm_sst_all.isel(nEns=iens)

    setname = 'LGMR_SST_lgm_member' + memstring
    ds_usingthis = ds_lgm_sst ## choose whether to do holo or lgm



    ## set lgm_sst_climo to desired file

    ## ensemble member version
    lgm_sst_climo = ds_usingthis.to_dataset()
    lgm_sst_climo['sst'] = lgm_sst_climo.sst_ann

    lgm_sst_climo['mask'] = xr.where(~np.isnan(lgm_sst_climo.sst), 1, 0)

    #############
    ## Load SST desired grid, using old lgmDA as a ref

    ddir = '/home/disk/atmos/vcooper/work/p2c2/lgm/'
    dfile = 'lgmDA_hol_SST_monthly_climo_v2.1.nc'
    ncf = ddir + dfile
    tempds = xr.open_dataset(ncf)
    tempds = tempds.set_coords(['lat','lon','month'])
    # tempds = xr.merge([tempds.set_coords(['lat','lon','month','ens']).sst,
    #                    tempds.set_coords(['lat','lon','month','ens']).ens])
    holo_sst_climo = tempds.assign_coords(month=('nmonth',np.arange(12)+1))
    holo_sst_climo['mask'] = xr.where(~np.isnan(holo_sst_climo.sst.isel(nmonth=0)), 1, 0)


    ################
    # xesmf regridding 
    newgrid = holo_sst_climo # desired grid
    data_for_regridding = lgm_sst_climo

    regridder = xe.Regridder(data_for_regridding, newgrid,
                             method='bilinear',
                             periodic=True,
                             extrap_method='inverse_dist',
                             filename='bilinear_LGMRens_extrapID.nc',
                             reuse_weights=True)
    lgm_sst_climo_hologrid_extrap = regridder(lgm_sst_climo)

    ##### BEGIN INFILLING ###
    # for msel in range(12): ## NOT DOING MONTHLY
    #     msel=0
    ds = lgm_sst_climo.sst#[msel]#.sel(nLat=latslice,nLon=lonslice)
    ds_holo = holo_sst_climo.sst[0] #[msel]#.sel(nLat=latslice,nLon=lonslice) #DUMMY
    data_all = ds.values.ravel()

    # llon, llat = np.meshgrid(ds.lon, ds.lat) ## extra lines because of regular grid
    # xc = llon.ravel()[~np.isnan(data_all)]
    # yc = llat.ravel()[~np.isnan(data_all)]
    # all_lons = llon.ravel()
    # all_lats = llat.ravel()
    ## IRREGULAR GRID option, switched on for LGMR
    xc = ds.lon.values.ravel()[~np.isnan(data_all)]
    yc = ds.lat.values.ravel()[~np.isnan(data_all)]
    all_lons = ds.lon.values.ravel()
    all_lats = ds.lat.values.ravel()

    data = data_all[~np.isnan(data_all)] ## exclude land for the "obs"
    data_coords = np.moveaxis(np.vstack([yc,xc]),-1,0)
    all_coords = np.moveaxis(np.vstack([all_lats,all_lons]),-1,0)

    ## also get icefrac data to segment the combo
    # ds_ice = lgm_ice_climo.icefrac[msel]
    ## find minimum latitude in NH that has sea ice;
    ## only will use krigging south of this latitude; not using this anymore
    # icelat = np.abs(ds_ice.where(
    #     (ds_ice > 0.01) & (ds_ice.lat > 0),drop=True).lat).min().values
    # print('icelat ', np.round(icelat))

    ## determine all patch centers
    latstep = 20 ## 25 for MARGO
    clats1 = np.arange(-50,50+1,latstep)
    #     print(clats1)
    clats2 = clats1[0:-1]+latstep/2
    #     print(' ',clats2)

    lonstep=60 ## 90 for MARGO
    clons1 = np.arange(0,360-lonstep+1,lonstep)
    #     print(clons1)
    clons2 = clons1[0:-1]+lonstep/2
    #     print(' ',clons2)

    ## initialize dictionary to store kriging results and distances
    krigpatch_dict = {}
    alld_dict = {}

    ## get great circle distance around a patch
    ## set maximum distance for patch based on distance to diagonal patch center
    choose_center_lats = clats1
    choose_center_lons = clons1
    choose_offset_lats = clats2
    choose_offset_lons = clons2

    for i,latval in enumerate(choose_center_lats):
        krigpatch_dict[latval] = {}
        alld_dict[latval] = {}

    #     for j,lonval in enumerate(clons1):
        for j,lonval in enumerate(choose_center_lons):
            cpair = np.hstack([latval,lonval])[np.newaxis,:]
    #             print(i, j, cpair)

            i2 = i
            j2 = j
            if i > (len(choose_offset_lats)-1): ## note this only works when doing full loop
                i2 = i-1
            if j > (len(choose_offset_lons)-1):
                j2 = j-1


            cpair_offset = np.hstack([choose_offset_lats[i2],
                                      choose_offset_lons[j2]])[np.newaxis,:]

            alld = haversine_distances(
                np.deg2rad(cpair),np.deg2rad(all_coords)).squeeze() * 6371 #km

            tempd = haversine_distances(
                np.deg2rad(cpair),np.deg2rad(data_coords)).squeeze() * 6371 #km

            maxdist = haversine_distances(
                np.deg2rad(cpair),np.deg2rad(cpair_offset)).squeeze() * 6371 #km

    #             print('maxdist = ', np.round(maxdist), 'km')

            data_inside_patch_index = np.where(tempd < maxdist)[0]
    #             print('size of patch data: ',data_inside_patch_index.size)

            np.random.seed(1)
            ind = np.random.choice(data_inside_patch_index,
                                   size=np.min([2000,data_inside_patch_index.size]),replace=False)

            ## set points of observations for krig
            lat = data_coords[ind][:,0]
            lon = data_coords[ind][:,1]

            OK = pyok.OrdinaryKriging(
                lon,
                lat,
                data[ind],
                variogram_model="exponential", #spherical
                verbose=False,
                enable_plotting=False,
                coordinates_type="geographic",
            )

            ## set pairs for krig based on distance from central point
            krig_ind = np.where(alld < maxdist)[0]
            krig_lon = all_lons[krig_ind]
            krig_lat = all_lats[krig_ind]

            ## old grid for krig; alternative method
            grid_lon = np.linspace(0, 359, 480)
            grid_lat = np.linspace(-89.9, 89.9, 240)


            ## the actual kriging (slow part)
            field, s2 = OK.execute('points', krig_lon, krig_lat)

            ## reshape result to be on full grid
            field_pairs = np.zeros(all_coords[:,0].shape)
            field_pairs[:] = np.nan
            field_pairs[krig_ind] = field
            field_da = xr.DataArray(field_pairs.reshape(ds.shape),
                                    dims=ds.dims,coords=ds.coords)
            alld_da = xr.DataArray(alld.reshape(ds.shape),
                                    dims=ds.dims,coords=ds.coords)

            krigpatch_dict[latval][lonval] = xr.DataArray(field_pairs.reshape(ds.shape),
                                dims=ds.dims,coords=ds.coords)
            alld_dict[latval][lonval] = xr.DataArray(alld.reshape(ds.shape),
                                dims=ds.dims,coords=ds.coords)

    ## REPEAT PROCESS WITH LATS SWITCHED
    choose_center_lats = clats2
    choose_center_lons = clons2
    choose_offset_lats = clats1
    choose_offset_lons = clons1

    for i,latval in enumerate(choose_center_lats):
        krigpatch_dict[latval] = {}
        alld_dict[latval] = {}

    #     for j,lonval in enumerate(clons1):
        for j,lonval in enumerate(choose_center_lons):
            cpair = np.hstack([latval,lonval])[np.newaxis,:]
    #             print(i, j, cpair)

            i2 = i
            j2 = j
            if i > (len(choose_offset_lats)-1): ## note this only works when doing full loop
                i2 = i-1
            if j > (len(choose_offset_lons)-1):
                j2 = j-1


            cpair_offset = np.hstack([choose_offset_lats[i2],
                                      choose_offset_lons[j2]])[np.newaxis,:]

            alld = haversine_distances(
                np.deg2rad(cpair),np.deg2rad(all_coords)).squeeze() * 6371 #km

            tempd = haversine_distances(
                np.deg2rad(cpair),np.deg2rad(data_coords)).squeeze() * 6371 #km

            maxdist = haversine_distances(
                np.deg2rad(cpair),np.deg2rad(cpair_offset)).squeeze() * 6371 #km

    #             print('maxdist = ', np.round(maxdist), 'km')

            data_inside_patch_index = np.where(tempd < maxdist)[0]
    #             print('size of patch data: ',data_inside_patch_index.size)

            np.random.seed(1)
            ind = np.random.choice(data_inside_patch_index,
                                   size=np.min([2000,data_inside_patch_index.size]),replace=False)

            ## set points of observations for krig
            lat = data_coords[ind][:,0]
            lon = data_coords[ind][:,1]

            OK = pyok.OrdinaryKriging(
                lon,
                lat,
                data[ind],
                variogram_model="exponential", #spherical
                verbose=False,
                enable_plotting=False,
                coordinates_type="geographic",
            )

            ## set pairs for krig based on distance from central point
            krig_ind = np.where(alld < maxdist)[0]
            krig_lon = all_lons[krig_ind]
            krig_lat = all_lats[krig_ind]

            ## old grid for krig; alternative method
            grid_lon = np.linspace(0, 359, 480)
            grid_lat = np.linspace(-89.9, 89.9, 240)


            ## the actual kriging (slow part)
            field, s2 = OK.execute('points', krig_lon, krig_lat)

            ## reshape result to be on full grid
            field_pairs = np.zeros(all_coords[:,0].shape)
            field_pairs[:] = np.nan
            field_pairs[krig_ind] = field
            field_da = xr.DataArray(field_pairs.reshape(ds.shape),
                                    dims=ds.dims,coords=ds.coords)
            alld_da = xr.DataArray(alld.reshape(ds.shape),
                                    dims=ds.dims,coords=ds.coords)

            krigpatch_dict[latval][lonval] = xr.DataArray(field_pairs.reshape(ds.shape),
                                dims=ds.dims,coords=ds.coords)
            alld_dict[latval][lonval] = xr.DataArray(alld.reshape(ds.shape),
                                dims=ds.dims,coords=ds.coords)

            ### plot result
    #         proj = ccrs.Robinson()
    #         fig = plt.subplots(figsize=(5,4))
    #         ax = plt.subplot(projection=proj)
    #         plt.pcolormesh(field_da.lon,field_da.lat,field_da,
    #                        transform = ccrs.PlateCarree(),
    #                        cmap='plasma')
    #         plt.title(str(i) + ', ' + str(j) + ' ' + str(cpair.squeeze()))        
    #         plt.show()

    #         fig = plt.subplots(figsize=(5,4))
    #         ax = plt.subplot(projection=proj)
    #         plt.pcolormesh(field_da.lon,field_da.lat,1/alld_da,
    #                        transform = ccrs.PlateCarree(),
    #                        cmap='Reds',vmax=1/1000)
    #         plt.show()

    ### Addition for Bering Straight
    addl_cpairs = 0
    addl_cpairs +=1
    latval = 66
    lonval = 192
    krigpatch_dict[latval] = {}
    alld_dict[latval] = {}


    cpair = np.hstack([latval,lonval])[np.newaxis,:]
    print(i, j, cpair)

    alld = haversine_distances(
        np.deg2rad(cpair),np.deg2rad(all_coords)).squeeze() * 6371 #km

    tempd = haversine_distances(
        np.deg2rad(cpair),np.deg2rad(data_coords)).squeeze() * 6371 #km

    maxdist = 3000 # 4500 for MARGO
    # print('maxdist = ', np.round(maxdist), 'km')

    data_inside_patch_index = np.where(tempd < maxdist)[0]
    # print('size of patch data: ',data_inside_patch_index.size)

    np.random.seed(1)
    ind = np.random.choice(data_inside_patch_index,
                           size=np.min([1200,data_inside_patch_index.size]),replace=False)

    ## set points of observations for krig
    lat = data_coords[ind][:,0]
    lon = data_coords[ind][:,1]

    OK = pyok.OrdinaryKriging(
        lon,
        lat,
        data[ind],
        variogram_model="exponential", #spherical
        verbose=False,
        enable_plotting=False,
        coordinates_type="geographic",
    )

    ## set pairs for krig based on distance from central point
    krig_ind = np.where(alld < maxdist)[0]
    krig_lon = all_lons[krig_ind]
    krig_lat = all_lats[krig_ind]

    ## old grid for krig; alternative method
    grid_lon = np.linspace(0, 359, 480)
    grid_lat = np.linspace(-89.9, 89.9, 240)


    ## the actual kriging (slow part)
    field, s2 = OK.execute('points', krig_lon, krig_lat)

    ## reshape result to be on full grid
    field_pairs = np.zeros(all_coords[:,0].shape)
    field_pairs[:] = np.nan
    field_pairs[krig_ind] = field
    field_da = xr.DataArray(field_pairs.reshape(ds.shape),
                            dims=ds.dims,coords=ds.coords)
    alld_da = xr.DataArray(alld.reshape(ds.shape),
                            dims=ds.dims,coords=ds.coords)

    krigpatch_dict[latval][lonval] = xr.DataArray(field_pairs.reshape(ds.shape),
                        dims=ds.dims,coords=ds.coords)
    alld_dict[latval][lonval] = xr.DataArray(alld.reshape(ds.shape),
                        dims=ds.dims,coords=ds.coords)

    weights = np.zeros(np.append(ds.shape,
                             len(clats1)*len(clats1) + len(clats2)*len(clats2) + addl_cpairs))

    l = 0
    for i,val in alld_dict.items():
        for j,val2 in val.items():
            weights[:,:,l] = 1/(val2+1) * (krigpatch_dict[i][j]/krigpatch_dict[i][j])
            l += 1

    wsum = np.nansum(weights,axis=2)
    wsum[wsum == 0] = np.nan

    frank_weighted = np.zeros(ds.shape)

    l = 0
    for i,val in krigpatch_dict.items():
        for j,da in val.items():
            frank_weighted += np.nan_to_num(da * weights[:,:,l])
            l += 1

    frank_weighted = frank_weighted / wsum

    ocean_ind = ds_holo/ds_holo 
    frank_mask = ocean_ind * frank_weighted

    # create a unique mask based on where kriging is necessary vs. extrap
    # this decision is based on inspection
    # SKIP THIS STEP FOR MARGO
    temp_mask = frank_mask.where((frank_mask.nLon > 0) & 
                 (frank_mask.nLat < 280) & 
                 (frank_mask.nLon < 40) | 
                 (frank_mask.nLon >= 40) & 
                 (frank_mask.nLon < 140) & 
                 (frank_mask.nLat < 278) | 
                 (frank_mask.nLon >= 140) & 
                 (frank_mask.lat < 69) & 
                 (frank_mask.nLon < 210) | 
                 (frank_mask.nLon >= 210) & 
                 (frank_mask.nLat < 325) & 
                 (frank_mask.nLon < 293) | 
                 (frank_mask.nLon >= 293) & 
                 (frank_mask.lat < 58))
    temp_mask = temp_mask.where(temp_mask.nLat > 40)

    ## combine the kriging with the lgm original
    krig_wlgm = xr.where(np.isnan(ds) & ~np.isnan(temp_mask.values),frank_mask.values,ds)

    ## regrid this combo onto holo grid with extrapolation to fill in high lats
    newgrid = holo_sst_climo.isel(nmonth=1) # desired grid
    data_for_regridding = krig_wlgm.to_dataset()
    data_for_regridding['mask'] = xr.where(~np.isnan(data_for_regridding.sst),1,0)

    regridder = xe.Regridder(data_for_regridding, newgrid,
                             method='bilinear',
                             periodic=True,extrap_num_src_pnts=64,
                             extrap_method='inverse_dist',
                             filename='bilinear_krigwlgm_to_holo_per_extrapID64.nc',
                             reuse_weights=True)

    merged = regridder(krig_wlgm).to_dataset()
    merged['mask'] = xr.where(
        ~np.isnan(merged.sst),1,0)

    ## save files
    # mstring = str(frank_mask.month.values).zfill(2)
    savepath = '/home/disk/sipn/vcooper/nobackup/lgm/infilled/ensLGMR/'
    # fname_krig = 'LGMR_lgmNorm_SST_monthly_climo_krigged_' + mstring + '.nc'
    fname_merged = setname + '_merged.nc'

    # frank_mask.to_netcdf(savepath + fname_krig)
    merged.to_netcdf(savepath + fname_merged)
    print('finished saving mem ' + memstring)
    print(datetime.datetime.now() - begin_time)