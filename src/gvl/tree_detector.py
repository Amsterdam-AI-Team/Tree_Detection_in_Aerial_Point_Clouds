# Based on https://www.linkedin.com/pulse/bomen-herkennen-een-3d-puntenwolk-arno-timmer/

import time
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, sjoin
from hdbscan import HDBSCAN
from scipy.spatial.qhull import ConvexHull
from shapely.geometry import Point
from shapely.wkt import loads
from sklearn.cluster import KMeans

from .helper_functions import \
    find_n_clusters_peaks, \
    former_preprocess_now_add_pid


class DetectorTree:
    '''

    '''

    def __init__(self, points):
        """
        Initialize the tree class holding all the points
        :param box: a tuple of (xmin, xmax, ymin, ymax)

        """

        self.raw_points = points

        # Masks
        self.n_returns_mask = self.raw_points['NumberOfReturns'] != 1
        self.n_many_returns_mask = self.raw_points['ReturnNumber'] >= 1 # TODO not used

        self.masks = self.n_returns_mask # TODO

        # initialize an empty DataFrames for writing to database later
        self.tree_coords = pd.DataFrame(data={'X': [],
                                              'Y': [],
                                              'Z': []
                                              })
        self.tree_df = GeoDataFrame({'xy_clusterID': [],
                                     'geometry': [],
                                     "meanZ": [],
                                     "n_pts": [],
                                     "NZ": []})

    def hdbscan_on_points(self, min_cluster_size, min_samples, xyz=False):
        """
        Performs hdbscan on input points.
        :param min_cluster_size: [int], minimum cluster size
        :param min_samples: : [int], min samples
        >> see https://hdbscan.readthedocs.io/en/latest/parameter_selection.html

        :param xyz: [bool] if True the clustering will be done over xyz otherwise xy.

        :return: writes the points assigned with clusters to self.clustered_points
        """

        masked_points = self.raw_points[self.masks]
        start = time.time()
        if xyz:
            xy = np.array([masked_points['X'], masked_points['Y'], masked_points['Z']]).T
        else:
            xy = np.array([masked_points['X'], masked_points['Y']]).T

        xy_clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                               min_samples=min_samples)
        xy_clusterer.fit(xy)

        clustered_points = pd.DataFrame({'X': masked_points['X'],
                                         'Y': masked_points['Y'],
                                         'Z': masked_points['Z'],
                                         'HAG': masked_points['HAG'],
                                         'NormalZ': masked_points['NZ'],
                                         'Classification': xy_clusterer.labels_})
        # remove "noise" points
        self.clustered_points = clustered_points[clustered_points.Classification >= 0]
        end = time.time()
        print(f'found {np.unique(len(np.unique(self.clustered_points.Classification)))[0]} xy_clusters')
        print(f'clustering on xy took {round(end - start, 2)} seconds')

    def convex_hullify(self, points, kmean_pols=False):
        """
        Makes a 2d convex hull around around a set of points.

        :param points: [pd.DataFrame] The points to hullify around
        :param kmean_pols: [bool] switch for chosing between the kmean polygons hullifier or not

        :return: writes to polygons in self.tree_df (initialized in init)
        """

        # empties the tree.df if it exists
        try:
            self.tree_df.drop(self.tree_df.index, inplace=True)
        except:
            pass

        for name, group in points.groupby('Classification'):
            if group.shape[0] <= 3:
                # remove polygons that contain too little points to hullify around
                try:
                    points.drop(points.groupby('Classification').get_group(name).index)
                except:
                    pass
            else:
                # performs convexhull
                coords = np.array([group.X, group.Y]).T
                # :TODO can I do this better? Params?
                polygon = ConvexHull(coords)

                # build wkt string
                wkt = 'POLYGON (('
                for group_id in polygon.vertices:
                    x, y = polygon.points[group_id]
                    wkt += f'{x} {y},'
                # close the polygon
                firstx, firsty = polygon.points[polygon.vertices[0]]
                wkt = wkt + f'{firstx} {firsty}))'

                points = self.add_group_to_result(group, kmean_pols, name, points, wkt)

    def add_group_to_result(self, group, kmean_pols, name, points, wkt):
        # if there are less than 3 points per square meter; it's not a tree
        if (group.shape[0] / loads(wkt).area) <= 3:
            try:
                points = points.drop(points.groupby('Classification').get_group(name).index)
                msg = f'dropped {name} because less than 3 pts/m2'
            except:
                msg = 'FAILED ' + msg
                pass
            print(msg)

        # if the area is larger than 800 m2; it's not a tree
        elif kmean_pols and loads(wkt).area >= 800:
            try:
                points = points.drop(points.groupby('Classification').get_group(name).index)
                print(f'dropped {name} because polygon is too big')
            except:
                pass

        # If the tree is smaller than 2 meters, it's not a tree.
        elif group.HAG.max() < 2:
            try:
                points = points.drop(points.groupby('Classification').get_group(name).index)
                print(f'dropped {name} the so called tree is too small < 1.5 m')
            except:
                pass
        # If most of the points point up; it's not a tree.
        elif kmean_pols and group.NormalZ.mean() > 0.7:
            try:
                points = points.drop(points.groupby('Classification').get_group(name).index)
                print(f'dropped {name} the so called tree is too flat nz > 0.7')
            except:
                pass

        # If more selection criteria are wanted, they go here.

        else:
            # write to df
            self.tree_df.loc[len(self.tree_df)] = [int(name),
                                                   loads(wkt),
                                                   group.HAG.mean(),
                                                   group.shape[0],
                                                   group.NormalZ.mean()]
        return points

    def find_points_in_polygons(self, polygon_df):
        """
        For the kmean clustering, more points is better.
        Therefor the returnmask is not used.
        This function finds all the raw points within a polygon.

        :param polygon_df: [pd.DataFrame] DataFrame with polygons

        :return: writes to self.xy_grouped_points
        """
        # remove ground points
        cluster_points = self.raw_points # TODO 

        cluster_data = former_preprocess_now_add_pid(
            cluster_points[['X', 'Y', 'Z', 'ReturnNumber', 'NumberOfReturns',
                            'HAG', 'NZ']])
        xy = [Point(coords) for coords in zip(cluster_points['X'], cluster_points['Y'], cluster_points['Z'])]
        points_df = GeoDataFrame(cluster_data, geometry=xy)

        # find raw points without ground within the polygons.
        grouped_points = sjoin(points_df, polygon_df, how='left')
        grouped_points.rename(columns={'NZ_left': 'NormalZ'}, inplace=True)
        grouped_points['X'] = grouped_points.geometry.apply(lambda p: p.x)
        grouped_points['Y'] = grouped_points.geometry.apply(lambda p: p.y)

        # TODO no idea where the nans are coming from
        # dirty hack, hope not to many important points go missing
        print(f'removing {np.isnan(grouped_points.index_right).sum()} mystery nans <- merge_points_polygons')
        grouped_points = grouped_points[~np.isnan(grouped_points.index_right)]

        # remove noise
        print(f'Removed {np.array([grouped_points.xy_clusterID < 0]).sum()} noise points')
        grouped_points = grouped_points[grouped_points.xy_clusterID >= 0]
        self.xy_grouped_points = grouped_points.rename(columns={'index_right': 'polygon_id'})

        # # remove ground points
        # cluster_points = self.raw_points[self.n_many_returns_mask]
        #
        # cluster_data = former_preprocess_now_add_pid(
        #     cluster_points[['X', 'Y', 'Z', 'ReturnNumber', 'NumberOfReturns',
        #                     'HAG', 'NZ']])
        # xy = [Point(coords) for coords in zip(cluster_points['X'], cluster_points['Y'], cluster_points['Z'])]
        # points_df = GeoDataFrame(cluster_data, geometry=xy)
        #
        # # find raw points without ground within the polygons.
        # grouped_points = sjoin(points_df, polygon_df, how='left')
        # grouped_points['X'] = grouped_points.geometry.apply(lambda p: p.x)
        # grouped_points['Y'] = grouped_points.geometry.apply(lambda p: p.y)
        #
        # # TODO no idea where the nans are coming from
        # # dirty hack, hope not to many important points go missing
        # print(f'removing {np.isnan(grouped_points.index_right).sum()} mystery nans <- merge_points_polygons')
        # grouped_points = grouped_points[~np.isnan(grouped_points.index_right)]
        #
        # # remove noise
        # print(f'Removed {np.array([grouped_points.xy_clusterID < 0]).sum()} noise points')
        # grouped_points = grouped_points[grouped_points.xy_clusterID >= 0]
        # self.xy_grouped_points = grouped_points.rename(columns={'index_right': 'polygon_id'})

    def kmean_cluster(self, xy_grouped_points, round_val, min_dist):
        """

        :param xy_grouped_points: [GeoPandasDataFrame]: the points classified per polygon
        :param min_dist: [int]: see find_n_clusters_peaks in self.kmean_cluster_group
        :param relative_threshold: [int]: see find_n_clusters_peaks in self.kmean_cluster_group
        :param round_val: [int]: see find_n_clusters_peaks in self.kmean_cluster_group

        :return: writes to self.kmean_grouped_points
        """
        # Initialize dataframes to write to
        to_cluster = pd.DataFrame(data={'pid': []})
        labs = pd.DataFrame(data={'labs': [0] * len(xy_grouped_points.pid),
                                  'pid': xy_grouped_points.pid,
                                  'HeightAboveGround': [0] * len(xy_grouped_points.pid),
                                  'NZ': [0] * len(xy_grouped_points.pid)},

                            index=xy_grouped_points.pid)
        self.kmean_grouped_points = xy_grouped_points.copy()

        for name, group in self.kmean_grouped_points.groupby('xy_clusterID'):
            tree_area = float(self.tree_df.loc[self.tree_df['xy_clusterID'] == int(name)].geometry.area)
            # to ensure no deep copy/copy stuff
            try:
                del new_labs
            except Exception as E:
                pass

            # A tree needs to be bigger than 2 meters^2 and have more than 10 points
            if name >= 0 and tree_area >= 2 and group.shape[0] >= 50:
                group = group.drop(['geometry'], axis=1)
                # run through second pdal filter
                # to_cluster = self.second_filter(group.to_records())
                to_cluster = group.copy()

                # actual kmeans clustering.
                kmeans_labels = self.kmean_cluster_group(group=to_cluster,
                                                         name=name,
                                                         round_val=round_val,
                                                         min_dist=min_dist)

                # Create the new labs dataframe
                new_labs = pd.DataFrame(data={'labs': kmeans_labels,
                                              'pid': to_cluster.pid,
                                              'HeightAboveGround': to_cluster.HAG,
                                              'NormalZ': to_cluster.NormalZ},
                                        index=to_cluster.pid)

                # add the new labels to the labels dataframe
                try:
                    labs.update(new_labs)
                except ValueError as e:
                    print(f'Fatal: {e}')
                    raise

                print(
                    f"polygon: {int(name)}  \t "
                    f"area:  {round(tree_area, 2)} \t "
                    f"Found {len(np.unique(kmeans_labels))} clusters"
                )

            else:
                new_labs = pd.DataFrame(data={'labs': [-1] * len(group.pid),
                                              'pid': to_cluster.pid},
                                        index=group.pid)
                labs.update(new_labs)
        self.kmean_grouped_points['value_clusterID'] = labs.labs * 10

        # Add columns for adding to the polygons later
        added_cols = ['HeightAboveGround', 'NZ']
        for col in added_cols:
            self.kmean_grouped_points[col] = eval(f'labs.{col}')

        # factorize the cluster labels
        combi_ids = ["".join(row) for row in
                     self.kmean_grouped_points[['value_clusterID', 'xy_clusterID']].values.astype(str)
                     if '-1' not in row]
        self.kmean_grouped_points['Classification'] = pd.factorize(combi_ids)[0]

    def kmean_cluster_group(self, group, name, round_val, min_dist):
        """
        Kmeans clustering performed on a subset (tree or cluster of trees) of points

        :param name: the cluster id of the group
        :param group: [pd.DataFrame]The points including x y and z columns
        :param min_dist: see find_n_clusters_peaks
        :param relative_threshold: see find_n_clusters_peaks
        :param round_val: see find_n_clusters_peaks
        :return: a series of the labels in the order of the input DataFrame
        """

        # Clustering is performed on only x y and z
        cluster_data = np.array([group.X,
                                 group.Y,
                                 group.Z]).T
        print(f'finding cluster centres for {name}')

        n_clusters, coordinates = \
            find_n_clusters_peaks(cluster_data=cluster_data
                                  # distance is rounded to a multiple of the gridsize
                                  , min_dist=min_dist
                                  , round_val=round_val
                                  )

        # to add initial cluster points
        if n_clusters > 1:
            coordinates = np.array(coordinates)
            # Add points to data
            self.tree_coords = self.tree_coords.append(
                pd.DataFrame(data={'X': coordinates.T[0],
                                   'Y': coordinates.T[1],
                                   'Z': coordinates.T[2]},
                             index=[name] * len(coordinates.T[0])))
            print(cluster_data.shape, coordinates.shape)
            # TODO max iter??
            kmeans = KMeans(n_clusters=n_clusters, max_iter=1000, init=np.array(coordinates), n_init=1).fit(
                cluster_data)
        else:
            # Add point to data
            print(f'n_clusters:: {n_clusters} \t '
                  f'n_pts = {cluster_data.shape[0]} \t '
                  f'{cluster_data.shape[0] / n_clusters} pts/clst')
            centroid = cluster_data.mean(axis=0)
            self.tree_coords = self.tree_coords.append(
                pd.DataFrame(data={'X': centroid[0],
                                   'Y': centroid[1],
                                   'Z': centroid[2]},
                             index=[name]))

            kmeans = KMeans(n_clusters=n_clusters, max_iter=1).fit(cluster_data)

        return kmeans.labels_

    def second_filter(self, points):
        out_points = points[points.HAG >= self.min_hag].copy()
        return pd.DataFrame(out_points)

    # def second_filter(self, points):
    #     """
    #     a second pdal filter. At the moment it is not used...
    #
    #     :param points: points on which to perform the filter
    #     :return: [pd.DataFrame] of the filtered points.
    #     """
    #     pipeline_config = {
    #         "pipeline": [
    #             {
    #                 "type": "filters.range",
    #                 "limits": "HAG[0.5:)"
    #             },
    #             {
    #                 "type": "filters.normal",
    #                 "knn": 8
    #             },
    #             {
    #                 "type": "filters.approximatecoplanar",
    #                 "knn": 8,
    #                 "thresh1": 25,
    #                 "thresh2": 6
    #             }
    #         ]
    #     }
    #
    #     try:
    #         p = pdal.Pipeline(json.dumps(pipeline_config), arrays=[points])
    #         p.validate()  # check if our JSON and options were good
    #         p.execute()
    #         # normalZ dissapears?
    #         arrays = p.arrays
    #         out_points = arrays[0]
    #
    #     except Exception as e:
    #         trace = traceback.format_exc()
    #         print(f'{points.shape[0]} points, probably not enough to second filter.')
    #         print("Unexpected error:", trace)
    #
    #         out_points = points.copy()
    #     return pd.DataFrame(out_points)
