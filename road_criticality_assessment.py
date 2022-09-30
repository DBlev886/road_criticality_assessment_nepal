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
from shapely.geometry import Point, LineString
import pycountry
from wpgpDownload.utils.wpcsv import Product
from wpgpDownload.utils.convenience_functions import download_country_covariates as dl
import requests
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from pysheds.grid import Grid
from rasterio.features import rasterize
import requests as r
import getpass
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

dem_path = 'C:/Users/Danny/Desktop/EMMA/IV/Data/DEM/ASTGTMV003_N28E083_dem.tif'
dem_proj_tif = 'C:/Users/Danny/Desktop/EMMA/IV/Data/DEM/ASTGTMV003_N28E083_dem_reproj.tif'


# %% set dataframe width for visualization
pd.set_option('display.max_colwidth', 50)


# %% >>> SOCIO-ECONOMIC IMPORTANCE
# fetch whole network for defined study area
network_graph = ox.graph_from_place(study_area,
                                    network_type="drive",
                                    clean_periphery=True)

# project the data
network_graph_proj = ox.project_graph(network_graph)
all_nodes, all_edges = ox.graph_to_gdfs(network_graph_proj,
                                        nodes=True,
                                        edges=True)
project_crs = all_nodes.crs

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
remaining = all_edges[all_edges['highway'].isin(['trunk', 'primary', 'tertiary']) == False]


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
frames = [trunk_union, primary_union, tertiary_union]
main_roads_to_cut = pd.concat(frames)
main_roads_to_cut = main_roads_to_cut.reset_index()
remaining = remaining.reset_index()

main_roads_to_cut = main_roads_to_cut.geometry
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


# %% open downloaded raster, reproject, and mask with aoi boundary
raster_crs = CRS(aoi.crs).to_epsg()
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
nodes = gpd.sjoin(all_nodes, rem_buffered, how="inner", predicate="intersects")
nodes.drop('index_right',
           axis=1, inplace=True)

pop_nodes = gpd.sjoin(all_nodes, nodes, how="inner", predicate="intersects")
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

cut_network['road_pop'] = buffered_pop['road_pop']
cut_network = gpd.sjoin(cut_network, all_edges, how="left", predicate="intersects")

cut_network['norm_pop'] = (cut_network['road_pop'] - cut_network['road_pop'].min()) / (cut_network['road_pop'].max() - cut_network['road_pop'].min())
cut_network['si_quantile'] = cut_network['centrality_avg'] + cut_network['norm_pop']

# qcut: quantile-based discretization to assign scores from 1 to 5 to centrality values
cut_network['si_score'] = pd.qcut(cut_network['si_quantile'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)


# %% >>> footprints
gdf = ox.geometries_from_place(study_area,
                               tags={'building': True})
gdf_proj = ox.project_gdf(gdf)

gdf_centroid = gdf_proj.centroid
gdf_df = gpd.GeoDataFrame(geometry=gpd.GeoSeries(gdf_centroid))
west, south, east, north = aoi.unary_union.buffer(0.1).bounds


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
cut_network = cut_network.drop_duplicates(subset=['ID'], keep="first")
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


# %% >>> MULTI-HAZARD ASSESSMENT
# download DEM
user = getpass.getpass(prompt='vidal92')          # Input NASA Earthdata Login Username
password = getpass.getpass(prompt='nasaPW977')    # Input NASA Earthdata Login Password

api = 'https://urs.earthdata.nasa.gov/oauth/authorize?response_type=code&client_id=OLpAZlE4HqIOMr0TYqg7UQ&redirect_uri=https%3A%2F%2Fd53njncz5taqi.cloudfront.net%2Furs_callback&state=https%3A%2F%2Fsearch.earthdata.nasa.gov%2Fsearch%2F%3Fee%3Dprod'
token_response = r.post('{}login'.format(api), auth=(user, password)).json()  # Insert API URL, call login service, provide credentials & return json
del user, password                                                            # Remove user and password information
token_response                                                                # Print response
'''
token = token_response['token']                      # save login token to variable
head = {'Authorization': 'Bearer {}'.format(token)}  # create header to store token information, needed to submit request
'''


# %%
# grid = Grid.from_raster('C:/Users/Danny/Desktop/EMMA/IV/Data/DEM/ASTGTMV003_N28E083_dem.tif',
#                         window_crs=project_crs,
#                         nodata=0)
# dem = grid.read_raster('C:/Users/Danny/Desktop/EMMA/IV/Data/DEM/ASTGTMV003_N28E083_dem.tif',
#                         window_crs=project_crs,
#                         nodata=0)

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
                resampling=Resampling.nearest)


# %%
grid = Grid.from_raster(dem_proj_tif)
dem = grid.read_raster(dem_proj_tif)

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
# Load some sample data
# lose a bit of resolution, but this is a fairly large file, and this is only an example.
shape = 1000, 1000
transform = rio.transform.from_bounds(*cut_network['geometry'].total_bounds, *shape)
rasterize_rivernet = rasterize(
    [(shape, 1) for shape in cut_network['geometry']],
    out_shape=shape,
    transform=transform,
    fill=0,
    all_touched=True,
    dtype=rio.uint8)

with rio.open(
    'rasterized-results.tif', 'w',
    driver='GTiff',
    dtype=rio.uint8,
    count=1,
    width=shape[0],
    height=shape[1],
    transform=transform
) as dst:
    dst.write(rasterize_rivernet, indexes=1)
    dst.crs = CRS.from_epsg(raster_crs)


# %%
# Read points from shapefile
pts = all_nodes
pts.index = range(len(pts))
coords = [(x, y) for x, y in zip(pts.lon, pts.lat)]

# Open the raster and store metadata
src = rio.open(dem_path)

# Sample the raster at every point location and store values in DataFrame
pts['raster_value'] = [x[0] for x in src.sample(coords)]

lowest_point = pts.nsmallest(1, 'raster_value')


# %%
x = float(lowest_point['lon'])
y = float(lowest_point['lat'])

# Snap pour point to high accumulation cell
x_snap, y_snap = grid.snap_to_mask(acc > 1000, (x, y))

# Delineate the catchment
catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, xytype='coordinate')

# Plot the result
grid.clip_to(catch)
catch_view = grid.view(catch)

# Plot the catchment
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_alpha(0)

plt.grid('on', zorder=0)
im = ax.imshow(np.where(catch_view, catch_view, np.nan), extent=grid.extent,
               zorder=1, cmap='Greys_r')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Delineated Catchment', size=14)


# %%
# Extract river network
branches = grid.extract_river_network(fdir, acc > 100)

sns.set_palette('husl')
fig, ax = plt.subplots(figsize=(8.5, 6.5))

plt.xlim(grid.bbox[0], grid.bbox[2])
plt.ylim(grid.bbox[1], grid.bbox[3])
ax.set_aspect('equal')

for branch in branches['features']:
    line = np.asarray(branch['geometry']['coordinates'])
    plt.plot(line[:, 0], line[:, 1])

_ = plt.title('Channel network (>100 accumulation)', size=14)


# %% check significance of HAND
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


# %%
# Estimating inundation extent (constant channel depth)
inundation_extent = np.where(hand_view < 3, 3 - hand_view, np.nan)

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


# %% landcover raster
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
                                            0)
'''
