import numpy as np

from scipy.stats import binned_statistic_2d
from osgeo import gdal
from osgeo import osr


def get_tile_grid(points, hag, x_min, y_min, tilewidth, res):
    x_edge = np.arange(x_min, x_min+tilewidth+res, res)
    y_edge = np.arange(y_min, y_min+tilewidth+res, res)
    n_points = (binned_statistic_2d(x=points[:, 0],
                                    y=points[:, 1],
                                    values=None,
                                    bins=[x_edge, y_edge],
                                    statistic='count')
                ).statistic
    hag = (binned_statistic_2d(x=points[:, 0],
                               y=points[:, 1],
                               values=hag,
                               bins=[x_edge, y_edge],
                               statistic=lambda y: np.percentile(y, 80))
           ).statistic
    nap = (binned_statistic_2d(x=points[:, 0],
                               y=points[:, 1],
                               values=points[:, 2],
                               bins=[x_edge, y_edge],
                               statistic=lambda y: np.percentile(y, 80))
           ).statistic
    return n_points, hag, nap


def save_geotiff(fname, n_points, hag, nap, x_min, y_min, res, epsg=28992):
    x_dim, y_dim = n_points.shape

    # set geotransform
    geotransform = (x_min, res, 0, y_min, 0, res)

    # create the 3-band raster file
    geotiff = (gdal.GetDriverByName('GTiff')
               .Create(fname, x_dim, y_dim, 3, gdal.GDT_Float32))

    geotiff.SetGeoTransform(geotransform)     # specify coords
    srs = osr.SpatialReference()              # establish encoding
    srs.ImportFromEPSG(epsg)                  # set coordinate system
    geotiff.SetProjection(srs.ExportToWkt())  # export coords to file

    geotiff.GetRasterBand(1).WriteArray(n_points)
    geotiff.GetRasterBand(1).SetNoDataValue(0)
    geotiff.GetRasterBand(1).SetDescription('n_points')

    geotiff.GetRasterBand(2).WriteArray(hag)
    geotiff.GetRasterBand(2).SetNoDataValue(0)
    geotiff.GetRasterBand(2).SetDescription('hag')

    geotiff.GetRasterBand(3).WriteArray(nap)
    geotiff.GetRasterBand(3).SetNoDataValue(0)
    geotiff.GetRasterBand(3).SetDescription('nap')

    geotiff.FlushCache()                      # write to disk
    geotiff = None
