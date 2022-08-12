import numpy as np
from tifffile import TiffFile, imread
import zarr

from upcp.utils.ahn_utils import GeoTIFFReader, fill_gaps, smoothen_edges


class GeoTIFFReader2(GeoTIFFReader):
    """
    GeoTIFFReader for AHN3 data. The data folder should contain on or more 0.5m
    resolution GeoTIFF files with the original filename.
    Parameters
    ----------
    data_folder : str or Path
        Folder containing the GeoTIFF files.
    fill_gaps : bool (default: True)
        Whether to fill gaps in the AHN data.
    max_gap_size : int (default: 50)
        Max gap size for gap filling. Only used when fill_gaps=True.
    smoothen : bool (default: True)
        Whether to smoothen edges in the AHN data.
    smooth_thickness : int (default: 1)
        Thickness for edge smoothening. Only used when smoothen=True.
    """

    RESOLUTION = 5

    def __init__(self, data_folder,
                 fill_gaps=True, max_gap_size=50,
                 smoothen=True, smooth_thickness=1):
        super().__init__(data_folder, False, fill_gaps, max_gap_size,
                         smoothen, smooth_thickness)

    def _readfolder(self):
        """
        Read the contents of the folder. Internally, a DataFrame is created
        detailing the bounding boxes of each available file to help with the
        area extraction.
        """
        file_match = "M*.TIF"

        for file in self.path.glob(file_match):
            with TiffFile(file.as_posix()) as tiff:
                if not tiff.is_geotiff:
                    print(f'{file.as_posix()} is not a GeoTIFF file.')
                elif ((tiff.geotiff_metadata['ModelPixelScale'][0]
                       != self.RESOLUTION)
                      or (tiff.geotiff_metadata['ModelPixelScale'][1]
                          != self.RESOLUTION)):
                    print(f'{file.as_posix()} has incorrect resolution.')
                else:
                    (x, y) = tiff.geotiff_metadata['ModelTiepoint'][3:5]
                    (h, w) = tiff.pages[0].shape
                    x_min = x - self.RESOLUTION / 2
                    y_max = y + self.RESOLUTION / 2
                    x_max = x_min + w * self.RESOLUTION
                    y_min = y_max - h * self.RESOLUTION
                    self.ahn_df.loc[file.name] = [file.as_posix(),
                                                  x_min, y_max, x_max, y_min]
        if len(self.ahn_df) == 0:
            print(f'No GeoTIFF files found in {self.path.as_posix()}.')
        else:
            self.ahn_df.sort_values(by=['Xmin', 'Ymax'], inplace=True)

    def filter_area(self, bbox, fill_value=np.nan):
        """Extract an area from the GeoTIFF data."""
        (bx_min, by_min, bx_max, by_max) = bbox

        ahn_tile = {}

        # We first check if the entire area is within a single TIF tile.
        query_str = '''(Xmin <= @bx_min) & (Xmax >= @bx_max) \
                        & (Ymax >= @by_max) & (Ymin <= @by_min)'''
        target_frame = self.ahn_df.query(query_str)
        if len(target_frame) == 0:
            print('No data found for area.')
            return None
        elif len(target_frame) == 1:
            # The area is within a single TIF tile, so we can easily return the
            # array.
            [path, x, y, w, h] = target_frame.iloc[0].values
            with imread(path, aszarr=True) as store:
                z_data = np.array(zarr.open(store, mode="r"))
            x_start = int((bx_min - x) / self.RESOLUTION)
            x_end = int((bx_max - x) / self.RESOLUTION)
            y_start = int((y - by_max) / self.RESOLUTION)
            y_end = int((y - by_min) / self.RESOLUTION)
            ahn_tile['x'] = np.arange(bx_min + self.RESOLUTION / 2,
                                      bx_max, self.RESOLUTION)
            ahn_tile['y'] = np.arange(by_max - self.RESOLUTION / 2,
                                      by_min, -self.RESOLUTION)
            ahn_tile['ground_surface'] = z_data[y_start:y_end, x_start:x_end]
            fill_mask = ahn_tile['ground_surface'] > 1e5
            ahn_tile['ground_surface'][fill_mask] = fill_value
            if self.fill_gaps:
                fill_gaps(
                    ahn_tile, max_gap_size=self.max_gap_size, inplace=True)
            if self.smoothen:
                smoothen_edges(
                    ahn_tile, thickness=self.smooth_thickness, inplace=True)
            return ahn_tile
        else:
            print('Area spanning multiple GeoTIFF tiles, not implemented yet.')
            return None
