
## Setup

Install [MiniConda](https://www.anaconda.com/docs/getting-started/miniconda/install#macos-terminal-installer) and
create the Python environment:

```bash
conda env create -f environment.yaml
```

Then activate the environment and install the dependencies:

```bash
# activate the # environment
conda activate vitiligo-seg

# install the dependencies
pip install -r requirements.txt
```

## Training the Model

To train the model, you'll need a dataset of labeled images. You can use [labelstud.io](https://labelstud.io/) to label the images using the [Semantic Segment with Polygons](https://labelstud.io/templates/image_polygons) template.

After labeling, export the dataset using the "YOLO with Images" format and place it in the `datasets` folder.

For initial testing, a sample dataset is available at `datasets/vitiligo-poc` for training a basic segmentation model.

Modify `train.py` to specify the model data (or adjust hyperparameters) and run:


```bash
python train.py
```

After training completes, the model will be saved in the `vitiligo-poc/fold_n/weights` folder. The `vitiligo-poc/fold_n` folder also contains additional files and charts for analysis.

## Run the app

To segment an image, run `gradio app.py`


