# Based on https://www.linkedin.com/pulse/bomen-herkennen-een-3d-puntenwolk-arno-timmer/

from random import sample
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max


def round_to_val(a, round_val):
    """
    :param a: numpy array to round
    :param round_val: value to round to
    :return: rounded numpy array
    """
    return np.round(np.array(a, dtype=float) / round_val) * round_val


def find_n_clusters_peaks(cluster_data, round_val, min_dist):
    """
    finds the number of local maxima and their coordinates in a pointcloud.

    :param cluster_data: dattaframe with X Y and Z values
    :param round_val: the grid size of the raster to detect peaks in
    :param min_dist: minimal distance of the peaks
    :return: returns number of peaks and the coordinates of the peaks
    """
    img, minx, miny = interpolate_df(cluster_data, round_val)
    indices = peak_local_max(img, min_distance=min_dist)
    indices = [list(x) for x in set(tuple(b) for b in indices)]
    n_clusters = len(indices)

    mins = [[minx, miny, 0]] * n_clusters  # indices.shape[0]
    z = [img[i[0], i[1]] for i in indices]
    round_val_for_map = [round_val] * n_clusters
    mapped = map(add_vectors, zip(indices, mins, z, round_val_for_map))
    coordinates = [coord for coord in mapped]
    coordinates = [list(x) for x in set(tuple(b) for b in coordinates)]

    return max(1, n_clusters), coordinates


def add_vectors(vec):
    """
    utility for summing vectors

    :param vec: vectors to add. Should contain 3 values,
     coordinates, minima and z values
    :return: a vector of summed vectors
    """
    coords, mins, z, round_val = vec
    y, x = coords
    minx, miny, minz = mins
    return [minx + (x * round_val), miny + (y * round_val), z]


def interpolate_df(xyz_points, round_val):

    xyz_points = xyz_points.T
    xyz_points = pd.DataFrame({'X': xyz_points[0],
                               'Y': xyz_points[1],
                               'Z': xyz_points[2] ** 2})

    xyz_points['x_round'] = round_to_val(xyz_points.X, round_val)
    xyz_points['y_round'] = round_to_val(xyz_points.Y, round_val)

    binned_data = xyz_points.groupby(['x_round', 'y_round'], as_index=False).max()

    minx = min(binned_data.x_round)
    miny = min(binned_data.y_round)

    x_arr = binned_data.x_round - min(binned_data.x_round)
    y_arr = binned_data.y_round - min(binned_data.y_round)

    img_size_x = int(round(max(x_arr), 1))
    img_size_y = int(round(max(y_arr), 1))

    img = np.zeros([img_size_y + 1, img_size_x + 1])

    img[round_to_val(y_arr / round_val, 1).astype(np.int),
        round_to_val(x_arr / round_val, 1).astype(np.int)] = binned_data.Z

    return img, minx, miny


def former_preprocess_now_add_pid(points):
    f_pts = pd.DataFrame(points)
    f_pts['pid'] = f_pts.index
    return f_pts


def color_clusters(grouped_points):
    colors = get_colors(len(np.unique(grouped_points['Classification'])))
    output_dataframe = grouped_points[['pid',
                                       'X', 'Y', 'Z',
                                       'ReturnNumber', 'Classification']]
    for i, color in enumerate(['Red', 'Green', 'Blue']):
        col = output_dataframe.apply(lambda row: colors[int(row['Classification'])][i], axis=1)
        output_dataframe.loc[:, color] = col

    return output_dataframe


def get_colors(n):
    cols = 100 * [[0, 0, 0], [1, 0, 103], [213, 255, 0], [255, 0, 86], [158, 0, 142], [14, 76, 161], [255, 229, 2],
                  [0, 95, 57], [0, 255, 0], [149, 0, 58], [255, 147, 126], [164, 36, 0], [0, 21, 68], [145, 208, 203],
                  [98, 14, 0], [107, 104, 130], [0, 0, 255], [0, 125, 181], [106, 130, 108], [0, 174, 126],
                  [194, 140, 159], [190, 153, 112], [0, 143, 156], [95, 173, 78], [255, 0, 0], [255, 0, 246],
                  [255, 2, 157], [104, 61, 59], [255, 116, 163], [150, 138, 232], [152, 255, 82], [167, 87, 64],
                  [1, 255, 254], [255, 238, 232], [254, 137, 0], [189, 198, 255], [1, 208, 255], [187, 136, 0],
                  [117, 68, 177], [165, 255, 210], [255, 166, 254], [119, 77, 0], [122, 71, 130], [38, 52, 0],
                  [0, 71, 84], [67, 0, 44], [181, 0, 255], [255, 177, 103], [255, 219, 102], [144, 251, 146],
                  [126, 45, 210], [189, 211, 147], [229, 111, 254], [222, 255, 116], [0, 255, 120], [0, 155, 255],
                  [0, 100, 1], [0, 118, 255], [133, 169, 0], [0, 185, 23], [120, 130, 49], [0, 255, 198],
                  [255, 110, 65], [232, 94, 190]]

    return sample(cols, n)
