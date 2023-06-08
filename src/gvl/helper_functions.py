import numpy as np
import re
import os
import pathlib
import laspy
import open3d as o3d
import shapely.geometry as sg
from upcp.preprocessing import ahn_preprocessing


DEFAULT_BOX_SIZE = 1000


def roundup(x, N=DEFAULT_BOX_SIZE):
    return x - x % -N


def rounddown(x, N=DEFAULT_BOX_SIZE):
    return x - x % +N


def box_to_name(box, box_size):
    (x_min, y_min, _, _) = box.bounds
    return f'{x_min/box_size:.0f}_{y_min/box_size:.0f}'


def get_tilecode_from_filename(filename, n_digits=3):
    """Extract the tile code from a file name."""
    return re.match(fr'.*(\d{{{n_digits}}}_\d{{{n_digits}}}).*', filename)[1]


def get_bbox_from_tile_code(tile_code, padding=0,
                            width=DEFAULT_BOX_SIZE, height=DEFAULT_BOX_SIZE):
    """
    Get the <X,Y> bounding box for a given tile code. The tile code is assumed
    to represent the lower left corner of the tile.
    Parameters
    ----------
    tile_code : str
        The tile code, e.g. 2386_9702.
    padding : float
        Optional padding (in m) by which the bounding box will be extended.
    width : int (default: 50)
        The width of the tile.
    height : int (default: 50)
        The height of the tile.
    Returns
    -------
    tuple of tuples
        Bounding box with inverted y-axis: ((x_min, y_max), (x_max, y_min))
    """
    tile_split = tile_code.split('_')

    # The tile code of each tile is defined as
    # 'X-coordinaat/50'_'Y-coordinaat/50'
    x_min = int(tile_split[0]) * width
    y_min = int(tile_split[1]) * height

    return ((x_min - padding, y_min + height + padding),
            (x_min + height + padding, y_min - padding))


def process_ahn_las_tile(ahn_las_file, out_folder='', resolution=0.1):
    if type(ahn_las_file) == pathlib.PosixPath:
        ahn_las_file = ahn_las_file.as_posix()
    tile_code = get_tilecode_from_filename(ahn_las_file)

    if out_folder != '':
        pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)

    ((x_min, y_max), (x_max, y_min)) = get_bbox_from_tile_code(tile_code)

    ahn_las = laspy.read(ahn_las_file)

    # Create a grid with 0.1m resolution
    grid_y, grid_x = np.mgrid[y_max-resolution/2:y_min:-resolution,
                              x_min+resolution/2:x_max:resolution]

    # Methods for generating surfaces (grids)
    ground_surface = ahn_preprocessing._get_ahn_surface(
                                            ahn_las, grid_x, grid_y, 'idw',
                                            ahn_preprocessing.AHN_GROUND)
    building_surface = ahn_preprocessing._get_ahn_surface(
                                            ahn_las, grid_x, grid_y, 'max',
                                            ahn_preprocessing.AHN_BUILDING,
                                            max_dist=1.5)

    filename = os.path.join(out_folder, 'ahn_' + tile_code + '.npz')
    np.savez_compressed(filename,
                        x=grid_x[0, :],
                        y=grid_y[:, 0],
                        ground=ground_surface,
                        building=building_surface)
    return filename


def calculate_normals(points_xyz):
    object_pcd = o3d.geometry.PointCloud()
    points = np.stack((points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]),
                      axis=-1)
    object_pcd.points = o3d.utility.Vector3dVector(points)
    object_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,
                                                              max_nn=30))

    normals = np.matrix.round(np.array(object_pcd.normals), 2)

    return normals


def voxel_downsample(points_xyz, voxel_size):
    ndims = points_xyz.shape[1]
    if ndims == 2:
        points_xyz = np.stack((points_xyz[:, 0], points_xyz[:, 1],
                               np.zeros_like(points_xyz[:, 0])), axis=-1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    downpts = np.asarray(downpcd.points)
    if ndims == 2:
        return downpts[:, :2]
    else:
        return downpts


# From https://stackoverflow.com/a/52173616
# CC BY-SA 4.0
def get_wl_box(points):
    """ Get width and length of a cluster of points. """
    polygon = sg.Polygon(points[:, :2])

    # get the minimum bounding rectangle and zip coordinates into a list of
    # point-tuples
    mbr_points = list(zip(*polygon
                          .minimum_rotated_rectangle
                          .exterior.coords.xy))

    # calculate the length of each side of the minimum bounding rectangle
    mbr_lengths = [sg.LineString((mbr_points[i], mbr_points[i+1])).length
                   for i in range(len(mbr_points) - 1)]

    # get major/minor axis measurements
    minor_axis = min(mbr_lengths)
    major_axis = max(mbr_lengths)

    return minor_axis, major_axis
