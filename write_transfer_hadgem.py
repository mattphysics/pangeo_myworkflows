
import os
from dask_gateway import Gateway, GatewayCluster
gateway = Gateway()
if len(gateway.list_clusters())>0:
    cluster = gateway.connect(gateway.list_clusters()[0].name)
else:
    cluster = GatewayCluster()
    cluster.adapt(minimum=2, maximum=10)  # or cluster.scale(n) to a fixed size.
#     cluster.adapt(minimum=2, maximum=100)  # or cluster.scale(n) to a fixed size.
client = cluster.get_client()

import xarray as xr
import numpy as np
# xr.set_options(display_style='html')
import intake

cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
col = intake.open_esm_datastore(cat_url)


model = 'HadGEM3-GC31-LL' # HadGEM3-GC31-MM # UKESM1-0-LL

cat = col.search(institution_id='MOHC',table_id=['day','Oday','Eday','CFday'],variable_id=['psl','tos'])
cat_hadgem3 = cat.search(variable_id=['tos','psl'],experiment_id='historical',source_id=model,member_id='r4i1p1f3')


dset_dict = cat_hadgem3.to_dataset_dict(zarr_kwargs={'consolidated': True})

# areacella = xr.open_dataset('areacella_fx_%s_piControl_r1i1p1f1_gn.nc' % model)
# areacello = xr.open_dataset('areacello_Ofx_%s_piControl_r1i1p1f1_gn.nc' % model)

ds = xr.merge([
    dset_dict['CMIP.MOHC.%s.historical.day.gn' % model]['psl'],
    dset_dict['CMIP.MOHC.%s.historical.Oday.gn' % model]['tos'],
#     areacella,
#     areacello
]).isel(lat=slice(None,None,-1))


# years = np.unique(ds['time.year'].values)
years = np.arange(1982,2015)

for year in years:
    print(year)
    dsii = ds.sel(time=str(year)).load()
    outname = '%s_psl_tos_%i.nc' % (model,year)
    print(outname)
    
    dsii.to_netcdf(outname)
    dsii.close()
    
    s = os.system('rsync -azP %s aengenheyster@gateway.atm.ox.ac.uk:/network/group/aopp/oceans/LZ001_AENGENHEYSTER_OHUFLUX/data/cmip6/' % outname)
    if s==0:
        print('Transfer success')
    os.system('rm %s' % outname)