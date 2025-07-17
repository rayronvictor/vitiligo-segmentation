
## Setup

Install [MiniConda](https://www.anaconda.com/docs/getting-started/miniconda/install#macos-terminal-installer) and
create the python environment

```bash
conda env create -f environment.yaml
```

After that, activate the environment and install the dependencies:

```bash
# activate the env
conda activate vitiligo-seg

# install the dependencies
pip install -r requirements.txt
```

## Training the model

To train the model, you will need a dataset of labeled images. You can use the [labelstud.io](https://labelstud.io/)
to label the images using the [Semantic Segment with Polygons](https://labelstud.io/templates/image_polygons).

After labeling, export the dataset using "Yolo with Images" format and put the data in the `datasets` folder.

As a start, there is already a dataset `datasets/vitiligo-poc` to training a simple segmentation model.

Change the `train.py` to point to the model data (or to try other hyperparams) and run:

```bash
python train.py
```

After the training is finished, the model you be placed at `runs/segment/train/weights` folder. The `runs/segment/train`
folder also includes other interesting files and charts.

## Predict

To segment an image, adjust the `predict.py` to point to the trained model and the image that will be segmented and run:

```bash
python predict.py
```


