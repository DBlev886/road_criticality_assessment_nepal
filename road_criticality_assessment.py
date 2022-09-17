# -*- coding: utf-8 -*-
# Created on Sun May  8 12:04:30 2022

# @author: Danny Baldig
# title: "Automating criticality assessments of rural road networks in Nepal"

# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# venv\scripts\activate


# %% >>> inputs and environment setup
import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
import shapely
import rasterio as rio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterstats
from rasterstats import zonal_stats
from pyproj import CRS
import numpy as np
from shapely import speedups
from shapely.geometry import Point, LineString, Polygon
from shapely import geometry, ops
import pycountry
from wpgpDownload.utils.wpcsv import Product
from wpgpDownload.utils.convenience_functions import download_country_covariates as dl
import requests
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from pysheds.grid import Grid
import pysheds
from rasterio.features import rasterize

import libpysal
import pydriosm
from shapely.validation import make_valid
import jenkspy

speedups.enabled

# path for (temporary) downloads
download_path = 'C:/Users/Danny/Desktop/EMMA/IV/Data/'
# path of land cover raster >>> AUTOMATIZE LATER
lc_path = "C:/Users/Danny/Desktop/EMMA/IV/Data/Land_cover/data/LandCover_NP_2019.tif"
# define study area according to OSM Nominatim (comma-seperated strings for
# several selection)s: https://nominatim.openstreetmap.org/ui/search.html
study_area = (["Beni, Nepal", "Jaljala, Nepal"])
# "constrained": for official population counts or
# "UNadj_constrained" for UN adjusted population counts
population_type = ""

dem_path = 'C:/Users/Danny/Desktop/EMMA/IV/Data/DEM/ASTGTMV003_NEPAL.tif'
dem_proj_tif = 'C:/Users/Danny/Desktop/EMMA/IV/Data/DEM/ASTGTMV003_NEPAL_proj.tif'
dem_temp = 'C:/Users/Danny/Desktop/EMMA/IV/Data/DEM/ASTGTMV003_temporary.tif'


# %% set dataframe width for visualization
pd.set_option('display.max_colwidth', 200)


# %% >>> SOCIO-ECONOMIC IMPORTANCE
# fetch whole network for defined study area
network_graph = ox.graph_from_place(study_area,
                                    network_type="drive",
                                    buffer_dist=2000,
                                    clean_periphery=True)

# project the data
network_graph_proj = ox.project_graph(network_graph)
all_nodes, all_edges = ox.graph_to_gdfs(network_graph_proj,
                                        nodes=True,
                                        edges=True)
project_crs = all_nodes.crs

all_nodes.to_file('nodes.shp')
'''def unique_list(a_list):
    seen = set()
    for x in a_list:
        key = repr(x)
        if key not in seen:
            seen.add(key)
            print(x)


unique_list(all_edges['highway'])
# print(all_edges['highway'].unique())'''


# %%
'''cf = '["highway"~"trunk|primary|tertiary"]'
main_network = ox.graph_from_place(study_area,
                                   simplify=False,
                                   custom_filter=cf)
# fig, ax = ox.plot_graph(G)

main_network_proj = ox.project_graph(main_network)
main_nodes, main_edges = ox.graph_to_gdfs(main_network_proj,
                                          nodes=True,
                                          edges=True)'''


# %% filter for each main road type
trunk = all_edges[all_edges['highway'].isin(['trunk'])]
primary = all_edges[all_edges['highway'].isin(['primary'])]
tertiary = all_edges[all_edges['highway'].isin(['tertiary'])]
remaining = all_edges[all_edges['highway'].isin(['motorway', 'trunk', 'primary']) == False]

study_roads = all_edges[all_edges['highway'].isin(['motorway', 'trunk', 'primary'])]


# %% merge edges to one linestring
trunk_lines = []
trunk_union = trunk.unary_union
trunk_union = shapely.ops.linemerge(trunk_union)
try:
    for line in trunk_union.geoms:
        trunk_lines.append(line)
        trunk_union = gpd.GeoDataFrame(crs=project_crs, geometry=[trunk_lines])
except:
    trunk_union = gpd.GeoDataFrame(crs=project_crs, geometry=[trunk_union])

primary_lines = []
primary_union = primary.unary_union
primary_union = shapely.ops.linemerge(primary_union)
try:
    for line in primary_union.geoms:
        primary_lines.append(line)
        primary_union = gpd.GeoDataFrame(crs=project_crs, geometry=[primary_lines])
except:
    primary_union = gpd.GeoDataFrame(crs=project_crs, geometry=[primary_union])

tertiary_lines = []
tertiary_union = tertiary.unary_union
tertiary_union = shapely.ops.linemerge(tertiary_union)
try:
    for line in tertiary_union.geoms:
        tertiary_lines.append(line)
        tertiary_union = gpd.GeoDataFrame(crs=project_crs, geometry=tertiary_lines)
except:
    tertiary_union = gpd.GeoDataFrame(crs=project_crs, geometry=tertiary_union)

#%%
study_roads_lines = []
study_roads_union = study_roads.unary_union
study_roads_union = shapely.ops.linemerge(study_roads_union)

d = {'geometry': [study_roads_union]}
study_roads_union = gpd.GeoDataFrame(d, crs=project_crs)


#%%
'''try:
    for line in study_roads_union.geoms:
        study_roads_lines.append(line)
        study_roads_union = gpd.GeoDataFrame(crs=project_crs, geometry=study_roads_lines)
except:
    study_roads_union = gpd.GeoDataFrame(crs=project_crs, geometry=study_roads_union)'''


# %%
# create spatial weights matrix
W = libpysal.weights.Queen.from_dataframe(remaining)
# get component labels
components = W.component_labels
remaining = remaining.dissolve(by=components)


# %% define cut-function
def cut(line, distance):
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [
                LineString(coords[:i + 1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]


def MultiCut(line, pieces):
    # define empty lists to append filtered lines and lines to cut
    lines_result = []
    lines_to_cut = []
    # ensure that there are at least two pieces
    if pieces == 1:  # result equals same line if one pieces
        lines_result.append(line)
    elif pieces == 2:  # cut line by middle point if two pieces
        distance = (line.length) / pieces
        lines_result.append(cut(line, distance)[0])
        lines_result.append(cut(line, distance)[1])
    else:  # cut line in number of pieces if more than three pieces
        # loop  from first to penultimate piece
        for i in range(1, pieces):
            # first piece is cut to save result and rest of line
            if i == 1:
                distance = (line.length) / pieces  # distance equals line lenght divided by number of pieces
                lines_result.append(cut(line, distance)[0])  # cut line and save first part in lines_result
                lines_to_cut = cut(line, distance)[1]  # save rest of line in lines_to_cut to continue with split

            # split line if pieces equal to pieces minus two; save only first part in lines_result
            # to continue with split
            if 1 < i <= pieces - 2:
                distance = (line.length) / pieces
                lines_result.append(cut(lines_to_cut, distance)[0])
                lines_to_cut = cut(lines_to_cut, distance)[1]

            # finally cut line if pieces  equal to pieces minus 1 and save both parts in lines_result
            if (i != 1) and (i == pieces - 1):
                distance = (line.length) / pieces
                lines_result.append(cut(lines_to_cut, distance)[0])
                lines_result.append(cut(lines_to_cut, distance)[1])

    return lines_result


# %%
'''
frames = [trunk_union, trunk_union]#, tertiary_union]
main_roads_to_cut = pd.concat(frames)
main_roads_to_cut = main_roads_to_cut.reset_index()
remaining = remaining.reset_index()

main_roads_to_cut = main_roads_to_cut.geometry'''

study_roads_union = study_roads_union.reset_index()
main_roads_to_cut = study_roads_union.geometry

road_results = []
for j in range(0, len(main_roads_to_cut)):
    if main_roads_to_cut[j].length > 1000:
        line = main_roads_to_cut[j]
        pieces = int(((line.length) // 1000) + 1)
        road_results.append(list(MultiCut(line, pieces)))


# %%
main_roads = gpd.GeoDataFrame(np.concatenate(road_results))
main_roads.columns = ['geometry']
main_roads = main_roads.set_crs(project_crs)


# %% >>> ASSIGN STRATEGICAL IMPORTANCE SCORES
# normalize centrality values to summarize them with population later to create quantiles
CC = nx.closeness_centrality(nx.line_graph(network_graph_proj))
all_edges['cc'] = pd.DataFrame.from_dict(CC, orient='index')
all_edges['norm_cc'] = (all_edges['cc'] - all_edges['cc'].min()) / (all_edges['cc'].max() - all_edges['cc'].min())  # not needed?

BC = nx.betweenness_centrality(nx.line_graph(network_graph_proj))  # weight="length")
all_edges['bc'] = pd.DataFrame.from_dict(BC, orient='index')
all_edges['norm_bc'] = (all_edges['bc'] - all_edges['bc'].min()) / (all_edges['bc'].max() - all_edges['bc'].min())  # not needed?

all_edges['centrality_avg'] = (all_edges['norm_cc'] + all_edges['norm_cc']) / 2
all_edges['norm_centrality'] = (all_edges['centrality_avg'] - all_edges['centrality_avg'].min()) / (all_edges['centrality_avg'].max() - all_edges['centrality_avg'].min())  # not needed?


# %%
true_rev = all_edges[all_edges['reversed'].isin([True])]
reversed = all_edges[all_edges['reversed'].isin([True, False]) == False]
reversed.sort_values(['length'], inplace=True, ascending=False)
reversed['length'] = reversed['length'].round(1)
reversed = reversed.drop_duplicates(subset=['length'], keep="first")
all_edges = true_rev.append(reversed)

all_edges.drop(['oneway',
                'reversed',
                'access',
                'lanes',
                'length',
                'maxspeed'],
               axis=1, inplace=True)
all_edges = all_edges.reset_index()


# >>> population raster
# identify ISO country code to automatically download population raster based on OSM-input
if not download_path.endswith("/"):
    download_path = download_path + "/"

study_area_str = " ".join(study_area)
for country in pycountry.countries:
    if country.name in study_area_str:
        iso_code = country.alpha_3

# %%
products = Product(iso_code)  # Where instead of NPL it could be any valid ISO code.
#  to list all the products for NPL
# for p in products:
#    if "2020" in p.dataset_name:
#        print('%s/%s\t%s\t%s' % (p.idx, p.country_name,p.dataset_name,p.path))

prod_name_input = "ppp_2020" + population_type

# dl(ISO=iso_code, out_folder=download_path, prod_name=prod_name_input)


# %% define boundaries to mask raster
aoi = ox.geocode_to_gdf(study_area)
aoi = aoi.dissolve()
aoi = aoi.to_crs(CRS(project_crs))
aoi['geometry'] = aoi.geometry.buffer(2000)
aoi_file = download_path + "aoi.shp"

bbox = aoi[aoi.geometry.map(lambda z: True if not z.is_empty else False)].geometry.map(lambda z: z.exterior.xy)


#%%
study_frame = gpd.read_file("C:/Users/Danny/Desktop/EMMA/IV/Data/Local_units/study_frame.shp")
study_frame = study_frame.dissolve()
study_bbox = study_frame.bounds


# %% open downloaded raster, reproject, and mask with aoi boundary
raster_crs = CRS(project_crs).to_epsg()
mask_coords = aoi['geometry']
file_name = iso_code + "_" + prod_name_input + ".tif"
file_name = file_name.lower()
pop_path = download_path + file_name
file_substr = ".tif"
pop_idx = pop_path.index(file_substr)
pop_proj_tif = pop_path[:pop_idx] + "_reproj" + pop_path[pop_idx:]
pop_proj_clip_tif = pop_proj_tif[:pop_idx] + "_clipped" + pop_proj_tif[pop_idx:]


# %%
with rio.open(pop_path, mode='r+') as pop:
    pop.nodata = 0
    transform, width, height = calculate_default_transform(
        pop.crs, raster_crs, pop.width, pop.height, *pop.bounds)
    kwargs = pop.meta.copy()
    kwargs.update({
        'crs': raster_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rio.open(pop_proj_tif, 'w', **kwargs) as pop_proj:
        pop_proj.nodata = 0
        for i in range(1, pop.count + 1):
            reproject(
                source=rio.band(pop, i),
                destination=rio.band(pop_proj, i),
                resampling=Resampling.nearest)


# %%
with rio.open(pop_proj_tif) as pop_proj:
    pop_out_image, out_transform = rio.mask.mask(pop_proj, aoi.geometry, crop=True)
    pop_out_meta = pop_proj.meta

pop_out_meta.update({"driver": "GTiff",
                     "height": pop_out_image.shape[1],
                     "width": pop_out_image.shape[2],
                     "transform": out_transform})

with rio.open(pop_proj_clip_tif, "w", **pop_out_meta) as pop_dest:
    pop_dest.nodata = 0
    pop_dest.write(pop_out_image)



# %% read clipped population raster and assign values to numpy nd array
pop_raster = rio.open(pop_proj_clip_tif)
pop_count_array = pop_raster.read(1)
affine = pop_raster.transform


# %% buffer edges
main_buffered = main_roads.buffer(50)
main_buffered = gpd.GeoDataFrame(geometry=gpd.GeoSeries(main_buffered))

rem_buffered = remaining.buffer(50)
rem_buffered = gpd.GeoDataFrame(geometry=gpd.GeoSeries(rem_buffered))


# %% calculating zonal statistics
main_pop_means = rasterstats.zonal_stats(main_buffered, pop_count_array,
                                         affine=affine,
                                         nodata=np.nan,
                                         stats=['mean'],
                                         geojson_out=True)

# extract average population data from list
main_pop_mean_list = []
i = 0

while i < len(main_pop_means):
    main_pop_mean_list.append(main_pop_means[i]['properties'])
    i = i + 1


# %% create df and assign scores based on quantiles ## get NaN values if not converting index
main_pop_mean = pd.DataFrame(main_pop_mean_list)
main_pop_mean = main_pop_mean.set_index(main_buffered.index)
main_buffered['road_pop'] = main_pop_mean['mean']


# %% calculating zonal statistics
rem_pop_means = rasterstats.zonal_stats(rem_buffered, pop_count_array,
                                        affine=affine,
                                        nodata=np.nan,
                                        stats=['mean'],
                                        geojson_out=True)

# extract average population data from list
rem_pop_mean_list = []
i = 0

while i < len(rem_pop_means):
    rem_pop_mean_list.append(rem_pop_means[i]['properties'])
    i = i + 1


# %% create df and assign scores based on quantiles ## get NaN values if not converting index
rem_pop_mean = pd.DataFrame(rem_pop_mean_list)
rem_pop_mean = rem_pop_mean.set_index(rem_buffered.index)
rem_buffered['int_pop'] = rem_pop_mean['mean']

buffered_pop = gpd.sjoin(main_roads, main_buffered, how="left", predicate="within")

rem_buffered_pop = gpd.sjoin(rem_buffered, remaining, how="inner", predicate="intersects")
rem_buffered_pop.drop('index_right', axis=1, inplace=True)
buffer_nodes = all_nodes.buffer(2)
buffer_nodes = gpd.GeoDataFrame(geometry=gpd.GeoSeries(buffer_nodes))


# %%
pop_nodes = gpd.sjoin(all_nodes, rem_buffered, how="inner", predicate="intersects")  # <___use rem_buffered_pop instead?
pop_nodes.drop('index_right',
               axis=1, inplace=True)


# %%
# create spatial weights matrix
W = libpysal.weights.Queen.from_dataframe(all_edges)
# get component labels
components = W.component_labels
all_roads_shape = all_edges.dissolve(by=components)

all_roads_shape.drop(all_roads_shape.columns.difference(['geometry']), 1, inplace=True)


# %%
cut_network = gpd.sjoin(main_roads, all_roads_shape, how="left", predicate="intersects")
cut_network['ID'] = cut_network.index + 1
buffered_pop = buffered_pop.reset_index(drop=True)
cut_network = cut_network.reset_index(drop=True)

cut_network.drop(['index_right'],
                 axis=1, inplace=True)

#%%
# calculate population along the road/1000m
cut_network['road_pop'] = buffered_pop['road_pop']
cut_network = gpd.sjoin(cut_network, all_edges, how="left", predicate="intersects")
cut_network['length'] = cut_network.length
cut_network['road_pop'] = cut_network['road_pop'] / cut_network['length'] * 1000

#%%
cut_network = cut_network.drop_duplicates(subset=['ID'], keep="first")
cut_network['norm_pop'] = (cut_network['road_pop'] - cut_network['road_pop'].min()) / (cut_network['road_pop'].max() - cut_network['road_pop'].min())
cut_network['si_quantile'] = cut_network['centrality_avg'] + cut_network['norm_pop']
#cut_network['si_quantile'] = cut_network['si_quantile'].astype(int)

cut_network['highway'] = cut_network['highway'].astype(str)
cut_network['osmid'] = cut_network['osmid'].astype(str)
cut_network['name'] = cut_network['name'].astype(str)

# qcut: quantile-based discretization to assign scores from 1 to 5 to centrality values
cut_network['si_score'] = pd.qcut(cut_network['si_quantile'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)


# %%
lc_shp = gpd.read_file("C:/Users/Danny/Desktop/EMMA/IV/Data/Land_cover/osm/shape/landuse.shp")
lc_shp = lc_shp.to_crs(CRS(project_crs))

lc_residential = lc_shp[lc_shp['type'].isin(['residential'])]
lc_residential.geometry = lc_residential.apply(lambda row: make_valid(row.geometry) if not row.geometry.is_valid else row.geometry, axis=1)
lc_residential.drop(['osm_id',
                     'name'],
                    axis=1, inplace=True)
lc_residential = lc_residential.dissolve().explode()
lc_residential = lc_residential.loc[(lc_residential.area >= 100000)]
lc_residential = lc_residential.reset_index()

lc_farmland = lc_shp[lc_shp['type'].isin(['farmland'])]
lc_farmland.geometry = lc_farmland.apply(lambda row: make_valid(row.geometry) if not row.geometry.is_valid else row.geometry, axis=1)
lc_farmland.drop(['osm_id',
                 'name'],
                 axis=1, inplace=True)
lc_farmland = lc_farmland.dissolve().explode()
lc_farmland = lc_farmland.loc[(lc_farmland.area >= 50000)]
lc_farmland = lc_farmland.reset_index()


# %%
cut_network.drop(['index_right'],
                 axis=1, inplace=True)
cut_network = gpd.sjoin(cut_network, lc_residential, predicate='intersects', how='left')

cut_network.rename(columns={'type': 'residential'}, inplace=True)
cut_network['residential'] = cut_network['residential'].replace(['residential'], '0.5').fillna('0').astype(np.float64)
cut_network.drop('index_right', axis=1, inplace=True)
cut_network = gpd.sjoin(cut_network, lc_farmland, predicate='intersects', how='left')
cut_network.rename(columns={'type': 'farmland'}, inplace=True)
cut_network['farmland'] = cut_network['farmland'].replace(['farmland'], '0.2').fillna('0').astype(np.float64)
cut_network.drop(['index_right',
                  'level_0_left',
                  'level_1_left',
                  'level_0_right',
                  'level_1_right'],
                 axis=1, inplace=True)

network_intersection = gpd.sjoin(cut_network, pop_nodes, predicate="intersects", how="left")
network_intersection['int_pop'].fillna('0')

pop_sum = network_intersection.groupby(['ID']).sum()
pop_sum = pop_sum[['int_pop']]
cut_network = cut_network.merge(pop_sum, on='ID')

# %%
breaks = jenkspy.jenks_breaks(cut_network['int_pop'], n_classes=3)
cut_network['pop_weight'] = pd.cut(cut_network['int_pop'],
                                   bins=breaks,
                                   labels=['0.1', '0.2', '0.3'])
cut_network['pop_weight'] = cut_network['pop_weight'].cat.add_categories('0').fillna('0')
cut_network['pop_weight'] = cut_network['pop_weight'].astype(np.float64)


# %%
cut_network['li_sum'] = cut_network['residential'] + cut_network['farmland'] + cut_network['pop_weight']

evaluation_bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
evaluation_labels = [1, 2, 3, 4, 5]

cut_network['li_score'] = pd.cut(cut_network['li_sum'], bins=evaluation_bins,
                            labels=evaluation_labels, include_lowest=True, right=False).astype(int)
# cut_network['li_score'] = pd.qcut(cut_network['li_quantile'], q=5, labels=[1, 2, 3, 4, 5]).astype(str)

cut_network['socioeconomic_score'] = (cut_network['li_score'] * 0.7) + (cut_network['si_score'] * 0.3)
cut_network['socioeconomic_score'] = pd.cut(cut_network['li_sum'], bins=evaluation_bins,
                            labels=evaluation_labels, include_lowest=True, right=False).astype(int)


# %%

cut_network['osmid'] = cut_network['osmid'].astype(str)
cut_network['name'] = cut_network['name'].astype(str)
cut_network.to_file('cut_network.shp')


# %%
'''cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]

rem_buffered_pop['highway'] = rem_buffered_pop['highway'].astype(str)
rem_buffered_pop['osmid'] = rem_buffered_pop['osmid'].astype(str)

for col in rem_buffered_pop.columns:
    if any(isinstance(val, list) for val in rem_buffered_pop[col]):
        print('Column: {0}, has a list in it'.format(col))
'''


# %% get first and last point for drainage calculation
cut_network['first'] = cut_network["geometry"].apply(lambda g: g.coords[0])
cut_network['last'] = cut_network["geometry"].apply(lambda g: g.coords[-1])



# %% landcover raster
'''from pyrosm import OSM, get_data

# Download a dataset for Nepal and update if already exists in temp
fp = get_data("Nepal", update=True)
osm = OSM(fp)

# Get AOI as bounding box
bounding_box = osm.get_boundaries(aoi)
bounding_box.plot()

# Get the shapely geometry from GeoDataFrame
bbox_geom = bounding_box['geometry'].values[0]

# Initiliaze with bounding box
osm = OSM(fp, bounding_box=bbox_geom)
# Retrieve buildings for AOI
aoi_landuse = osm.get_landuse()
# Let's plot the buildings and specify colors according the type of the building
ax = aoi_landuse.plot(column="building", figsize=(12,12), legend=True, legend_kwds=dict(loc='upper left', ncol=3, bbox_to_anchor=(1, 1)))
'''
# %%
'''
# trying to automatize land cover download /// doesn't work yet
site_url = 'http://rds.icimod.org/DatasetMasters/DownloadFile/19?metadataid=1972729'
userid = 'danny.baldig@student.uibk.ac.at'
password = 'icimodPW977'

file_url = 'http://rds.icimod.org/DatasetMasters/DownloadFile/19?metadataid=1972729'
o_file = 'nlcms_2019.zip'

# create session
s = requests.Session()
# GET request. This will generate cookie for you
s.get(site_url)
# login to site.
s.post(site_url, data={'_username': userid, '_password': password})
# Next thing will be to visit URL for file you would like to download.
lc_file = s.get(file_url)

# Download file
with open(o_file, 'wb') as output:
    output.write(lc_file.content)
print(f"requests:: File {o_file} downloaded successfully!")

# Close session once all work done
s.close()

lc_idx = lc_path.index(file_substr)
lc_proj_tif = lc_path[:lc_idx] + "_reproj" + lc_path[lc_idx:]
lc_proj_clip_tif = lc_proj_tif[:lc_idx] + "_clipped" + lc_proj_tif[lc_idx:]

with rio.open(lc_path) as lc:
    transform, width, height = calculate_default_transform(
        lc.crs, raster_crs, lc.width, lc.height, *lc.bounds)
    kwargs = lc.meta.copy()
    kwargs.update({
        'crs': raster_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rio.open(lc_proj_tif, 'w', **kwargs) as lc_proj:
        for i in range(1, lc.count + 1):
            reproject(
                source=rio.band(lc, i),
                destination=rio.band(lc_proj, i),
                resampling=Resampling.nearest)

with rio.open(lc_proj_tif) as lc_proj:
    lc_out_image, out_transform = rio.mask.mask(lc_proj, mask_coords, crop=True)
    lc_out_meta = lc_proj.meta

lc_out_meta.update({"driver": "GTiff",
                    "height": lc_out_image.shape[1],
                    "width": lc_out_image.shape[2],
                    "transform": out_transform})

# output with wrong CRS again
with rio.open(lc_proj_clip_tif, "w", **lc_out_meta) as lc_dest:
    lc_dest.write(lc_out_image)
    lc_dest.close()


# get zonal statistics to get predominant land covers
cut_network['land_cover_unique'] = zonal_stats(cut_network,
                                               lc_proj_clip_tif,
                                               categorical=True)

# land_cover codes: settlement: 6 + cropland: 7
# order important to assign higher value first
cut_network["settlement"] = np.select([cut_network["land_cover_unique"].astype(str).str.contains(r"6:")],
                                      [0.5],
                                      0)

cut_network["cultivated_land"] = np.select([cut_network["land_cover_unique"].astype(str).str.contains(r"7:")],
                                           [0.2],
                                           0)


cut_network["intersection"] = np.select([cut_network["land_cover_unique"].astype(str).str.contains(r"7:")],
                                            [0.3],
                                            0)'''


# %%
















import rasterio as rio
import getpass
import requests as r
import rio_terrain
import fiona


# %% >>> MULTI-HAZARD ASSESSMENT

'''grid = Grid.from_raster('C:/Users/Danny/Desktop/EMMA/IV/Data/DEM/ASTGTMV003_NEPAL.tif',
                         window_crs=project_crs,
                         nodata=0)
dem = grid.read_raster('C:/Users/Danny/Desktop/EMMA/IV/Data/DEM/ASTGTMV003_NEPAL.tif',
                         window_crs=project_crs,
                         nodata=0)

with rio.open(dem_path, mode='r+') as dem:
    transform, width, height = calculate_default_transform(
        dem.crs, raster_crs, dem.width, dem.height, *dem.bounds)
    kwargs = dem.meta.copy()
    kwargs.update({
        'crs': raster_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rio.open(dem_proj_tif, 'w', **kwargs) as dem_proj:
        for i in range(1, dem.count + 1):
            reproject(
                source=rio.band(dem, i),
                destination=rio.band(dem_proj, i),
                resampling=Resampling.nearest)'''


# %%


#%%
with rio.open(dem_path, mode='r+') as dem:
    dem.nodata = 0
    transform, width, height = calculate_default_transform(
        dem.crs, raster_crs, dem.width, dem.height, *dem.bounds)
    kwargs = dem.meta.copy()
    kwargs.update({
        'crs': raster_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rio.open(dem_proj_tif, 'w', **kwargs) as dem_proj:
        dem_proj.nodata = 0
        for i in range(1, dem.count + 1):
            reproject(
                source=rio.band(dem, i),
                destination=rio.band(dem_proj, i),
                resampling=Resampling.nearest)

#%%
buff = shapely.geometry.box(*aoi.total_bounds).buffer(25000, join_style=2)

with rio.open(dem_proj_tif) as dem_proj:
    dem_out_image, out_transform = rio.mask.mask(dem_proj, [buff], crop=True)
    dem_out_meta = dem_proj.meta

dem_out_meta.update({"driver": "GTiff",
                     "height": dem_out_image.shape[1],
                     "width": dem_out_image.shape[2],
                     "transform": out_transform})

with rio.open(dem_temp, "w", **dem_out_meta) as dem_dest:
    dem_dest.nodata = 0
    dem_dest.write(dem_out_image)


#%%
grid = Grid.from_raster(dem_temp)
dem = grid.read_raster(dem_temp)

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_alpha(0)

plt.imshow(dem, extent=grid.extent, cmap='terrain', zorder=1)
plt.colorbar(label='Elevation (m)')
plt.grid(zorder=0)
plt.title('Digital elevation map', size=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()


# %%
# Condition DEM
# ----------------------
# Fill pits in DEM
pit_filled_dem = grid.fill_pits(dem)
# Fill depressions in DEM
flooded_dem = grid.fill_depressions(pit_filled_dem)
# Resolve flats in DEM
inflated_dem = grid.resolve_flats(flooded_dem)


# %%
# Determine D8 flow directions from DEM
# ----------------------
# Specify directional mapping
dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

# Compute flow directions
# -------------------------------------
fdir = grid.flowdir(inflated_dem, dirmap=dirmap)

fig = plt.figure(figsize=(8, 6))
fig.patch.set_alpha(0)

plt.imshow(fdir, extent=grid.extent, cmap='viridis', zorder=2)
boundaries = ([0] + sorted(list(dirmap)))
plt.colorbar(boundaries=boundaries,
             values=sorted(dirmap))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Flow direction grid', size=14)
plt.grid(zorder=-1)
plt.tight_layout()


# %%
# Calculate flow accumulation
# --------------------------
acc = grid.accumulation(fdir, dirmap=dirmap)

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_alpha(0)
plt.grid('on', zorder=0)
im = ax.imshow(acc, extent=grid.extent, zorder=2,
               cmap='cubehelix',
               norm=colors.LogNorm(1, acc.max()),
               interpolation='bilinear')
plt.colorbar(im, ax=ax, label='Upstream Cells')
plt.title('Flow Accumulation', size=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()


# %%
# Extract river network
branches = grid.extract_river_network(fdir, acc > 500)

sns.set_palette('husl')
fig, ax = plt.subplots(figsize=(8.5, 6.5))

plt.xlim(grid.bbox[0], grid.bbox[2])
plt.ylim(grid.bbox[1], grid.bbox[3])
ax.set_aspect('equal')

for branch in branches['features']:
    line = np.asarray(branch['geometry']['coordinates'])
    plt.plot(line[:, 0], line[:, 1])

_ = plt.title('Channel network (>500 accumulation)', size=14)

branches_gdf = gpd.GeoDataFrame.from_features(branches, crs=project_crs)
# branches_gdf.to_file('branches.shp')


# %%
from shapely import wkt
cut_network['drainage_pt'] = cut_network.intersection(branches_gdf.unary_union)
cut_network['drainage_area'] = cut_network['drainage_pt']
cut_network['drainage_area'] = cut_network['drainage_area']
#cut_network['drainage_area'] = wkt.dumps(cut_network['drainage_area'])
cut_network['drainage_area'] = cut_network.drainage_area.apply(lambda x: wkt.dumps(x))


# %%
def slope_gradient(elev, cellsize):
    px, py = np.gradient(elev, cellsize)
    slope_percent = np.sqrt(px ** 2 + py ** 2)
    # convert percentage to degrees
    slope_deg = np.degrees(np.arctan(slope_percent))
    return slope_deg


#%%
catchment_area = []
catchment_slope = []
for row in cut_network.drainage_area:
    if "EMPTY" in row:
        catchment_area.append('0')
        catchment_slope.append('0')
    elif "MULTI" not in row:
        grid = Grid.from_raster(dem_temp)   # remove at the end
        dem = grid.read_raster(dem_temp)    # remove at the end
        temp = row.split(" ",1)[1]
        x, y = map(str, temp.strip('()').split(' '))
        x_snap, y_snap = grid.snap_to_mask(acc > 200, (x, y))

        # Delineate the catchment
        catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, xytype='coordinate')
        catch = grid.view(catch, dtype=np.uint8)

        # Plot the result
        grid.clip_to(catch)
        catch_view = grid.view(catch)

        # Create a vector representation of the catchment mask
        shapes = grid.polygonize(catch_view)
        catchment_polygon = ops.unary_union([geometry.shape(shape)
                                     for shape, value in shapes])
        catchment_area.append(catchment_polygon.area)

        # grid.clip_to(catch)
        dem_view = grid.view(dem)
        slope_deg = slope_gradient(dem_view, 30)
        mean_slope = np.mean(slope_deg)  #[slope_deg != 0])
        mean_slope = str(mean_slope).strip('()')
        catchment_slope.append(mean_slope)

    elif "MULTI" in row:
        multi_temp = row.split(" ", 1)[1]
        multi_list = multi_temp.strip('()').split(",")
        gdf = gpd.GeoDataFrame()
        for i in multi_list:
            grid = Grid.from_raster(dem_temp)   # remove at the end
            dem = grid.read_raster(dem_temp)    # remove at the end
            x, y = i.strip().split(' ')
            x_snap, y_snap = grid.snap_to_mask(acc > 200, (x, y))

            # Delineate the catchment
            catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, xytype='coordinate')
            catch = grid.view(catch, dtype=np.uint8)

            # Plot the result
            grid.clip_to(catch)
            catch_view = grid.view(catch)

            # Create a vector representation of the catchment mask
            shapes = grid.polygonize(catch_view)
            catchment_polygon = ops.unary_union([geometry.shape(shape)
                                        for shape, value in shapes])
            catch_gdf = gpd.GeoDataFrame({'geometry':[catchment_polygon]})
            #gdf = gdf.append(catch_gdf)
            gdf = pd.concat([gdf, catch_gdf])
            catchment_polygon = gdf.unary_union
            dem_view = grid.view(dem)
            slope_deg = slope_gradient(dem_view, 30)
            mean_slope = np.mean(slope_deg)  #[slope_deg != 0])
            mean_slope = str(mean_slope).strip('()')
        catchment_area.append(catchment_polygon.area)
        catchment_slope.append(mean_slope)
    else:
        catchment_area.append('Not_Rated')
        catchment_slope.append('Not_Rated')

km_conv = 1e4
catchment_area = [float(x) / km_conv for x in catchment_area]

#%%
def catch_area_score(x):
    if x < 50:
        return 1
    if 50 <= x < 100:
        return 2
    if 100 <= x < 500:
        return 3
    if 500 <= x < 1000:
        return 4
    if x >= 1000:
        return 5

def catch_slope_score(x):
    if x < 10:
        return 1
    if 10 <= x < 20:
        return 2
    if 20 <= x < 30:
        return 3
    if 30 <= x < 40:
        return 4
    if x >= 40:
        return 5

cut_network['catchment_area_[ha]'] = [round(item, 2) for item in catchment_area]
cut_network['catchment_slope [°]'] = [round(float(item), 2) for item in catchment_slope]
cut_network['catchment_area_score'] = cut_network['catchment_area_[ha]'].apply(catch_area_score)
cut_network['catchment_slope_score'] = cut_network['catchment_slope [°]'].apply(catch_slope_score)
cut_network


# %%
grid = Grid.from_raster(dem_temp)   # remove at the end
dem = grid.read_raster(dem_temp)    # remove at the end
temp = cut_network.drainage_area[1].split(" ",1)[1]
x, y = map(str, temp.strip('()').split(' '))
x_snap, y_snap = grid.snap_to_mask(acc > 200, (x, y))

# Delineate the catchment
catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, xytype='coordinate')
catch = grid.view(catch, dtype=np.uint8)
((catch >= 1)).sum()

# Plot the result
grid.clip_to(catch)
catch_view = grid.view(catch)

'''
# Plot the catchment
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_alpha(0)

plt.grid('on', zorder=0)
im = ax.imshow(np.where(catch_view, catch_view, np.nan), extent=grid.extent,
               zorder=1, cmap='Greys_r')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Delineated Catchment', size=14)'''

# # Create a vector representation of the catchment mask
shapes = grid.polygonize(catch_view)

catchment_polygon = ops.unary_union([geometry.shape(shape)
                                for shape, value in shapes])
catchment_polygon.area

#%%
# Create view
catch_view = grid.view(slope, dtype=np.float32)
# Create a vector representation of the catchment mask
shapes = grid.polygonize(catch_view)

# Specify schema
schema = {
        'geometry': 'Polygon',
        'properties': {'LABEL': 'float:16'}
}

# Write shapefile
with fiona.open('slope.shp', 'w',
                driver='ESRI Shapefile',
                crs=grid.crs.srs,
                schema=schema) as c:
    i = 0
    for shape, value in shapes:
        rec = {}
        rec['geometry'] = shape
        rec['properties'] = {'LABEL' : str(value)}
        rec['id'] = str(i)
        c.write(rec)
        i += 1


# %% check significance of HAND
'''
# Compute height above nearest drainage
hand = grid.compute_hand(fdir, dem, acc > 200)
# Create a view of HAND in the catchment
hand_view = grid.view(hand, nodata=np.nan)

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_alpha(0)
plt.imshow(hand_view,
           extent=grid.extent, cmap='terrain', zorder=1)
plt.colorbar(label='Height above nearest drainage (m)')
plt.grid(zorder=0)
plt.title('HAND', size=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()



# Estimating inundation extent (constant channel depth)
inundation_extent = np.where(hand_view < 300, 200 - hand_view, np.nan)
grid.viewfinder = fdir.viewfinder

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_alpha(0)
dem_view = grid.view(dem, nodata=np.nan)
plt.imshow(dem_view, extent=grid.extent, cmap='Greys', zorder=1)
plt.imshow(inundation_extent, extent=grid.extent,
           cmap='Blues', vmin=-5, vmax=10, zorder=2)
plt.grid(zorder=0)
plt.title('Inundation depths (constant channel depth)', size=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()


# Create a vector representation of the catchment mask
shapes = inundation_extent.polygonize(catch_view)

# Specify schema
schema = {
        'geometry': 'Polygon',
        'properties': {'LABEL': 'float:16'}
}

# Write shapefile
with fiona.open('catchment.shp', 'w',
                driver='ESRI Shapefile',
                crs=grid.crs.srs,
                schema=schema) as c:
    i = 0
    for shape, value in shapes:
        rec = {}
        rec['geometry'] = shape
        rec['properties'] = {'LABEL' : str(value)}
        rec['id'] = str(i)
        c.write(rec)
        i += 1


with rio.open(dem_path) as src:
    ras_data = src.read()
    ras_meta = src.profile

# make any necessary changes to raster properties, e.g.:
ras_meta['dtype'] = "int32"
ras_meta['nodata'] = -9999

with rio.open('inundation_extent.tif', 'w', **ras_meta) as dst:
    dst.write(inundation_extent, 1)
'''

# %%
