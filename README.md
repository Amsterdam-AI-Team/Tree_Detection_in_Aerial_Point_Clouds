# Tree Detection in Aerial Point Clouds

The [Actueel Hoogtebestand Nederland](https://www.ahn.nl/) (AHN) is the openly available elevation model of the Netherlands obtained by aerial laser scanning. 

| ![Point cloud](./media/examples/capture0.png) | ![Objects above ground](./media/examples/capture1.png)|![Static objects](./media/examples/capture2.png) | ![Static objects](./media/examples/capture3.png) |
|:---:|:---:|:---:|:---:|

<b>Example:</b> (left) AHN classification. (left-middle) filtered on AHN 'Other' class. (right-middle) Tree and unknown classes kept. (right) RandLA-Net classification. <br/>


---

## Folder Structure

 * [`media/examples`](./media/examples) _Visuals_
 * [`notebooks`](./notebooks) _Jupyter notebook tutorials_
 * [`src/gvl`](./src/upc_sw) _Python source code_

---

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/Amsterdam-AI-Team/Tree_Detection_in_Aerial_Point_Clouds.git
    ```

2. Install the dependencies:
    ```bash
    conda env create -f environment.yml
    ```

3. Additionally, install our [Urban PointCloud Processing](https://github.com/Amsterdam-AI-Team/Urban_PointCloud_Processing) package:
    ```bash
    pip install https://github.com/Amsterdam-AI-Team/Urban_PointCloud_Processing/releases/download/v0.1/upcp-0.1-py3-none-any.whl
    ```

4. Finally, install `cccorelib` and `pycc` by following the [instructions on their GitHub page](https://github.com/tmontaigu/CloudCompare-PythonPlugin/blob/master/docs/building.rst#building-as-independent-wheels). Please note, these two packages are not available on the Python Package Index (PyPi).

5. Check out the [notebooks](notebooks) for a demonstration.

---

## Usage

We provide tutorial [notebooks](notebooks) that demonstrate how the code can be used. This repository was designed to be used with the specific data source: AHN. AHN3 or AHN4 point clouds can be downloaded from [ArcGIS](https://www.arcgis.com/apps/Embed/index.html?appid=a3dfa5a818174aa787392e461c80f781) or [GeoTiles](https://geotiles.nl).

---

This repository was created by [Amsterdam Intelligence](https://amsterdamintelligence.com/) for the City of Amsterdam.
