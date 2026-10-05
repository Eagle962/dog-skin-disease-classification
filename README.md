# Dog Skin Disease Classification

Image classification of canine skin conditions (7 classes) with PyTorch (EfficientNet-B0, ImageNet-pretrained) and Grad-CAM visualisation.

Classes: `bacterial_dermatosis`, `dermatitis`, `dermatophytosis`, `flea_allergy_dermatitis`, `healthy`, `hypersensitivity_dermatitis`, `mange`

## Usage

```bash
pip install torch torchvision scikit-learn numpy pillow opencv-python-headless matplotlib tqdm pyyaml roboflow
export ROBOFLOW_API_KEY=<your key>
python3 merge_datasets.py            # download and merge datasets
python3 train_classification.py      # train and evaluate
python3 predict.py path/to/image.jpg # predict with Grad-CAM
```

## Data

Datasets are not included. `merge_datasets.py` downloads CC BY 4.0 datasets from Roboflow Universe:

- https://universe.roboflow.com/litespy-l22hu/dog-skin-diseases
- https://universe.roboflow.com/smartdog-f8su3/dog-skin-disease-dataset-bzhug
- https://universe.roboflow.com/research-on-health/gabungan-of798

Class labels were remapped and the data re-split. See [docs/開源平台佐證文件.md](docs/開源平台佐證文件.md) for the open-source platform documentation.

## Disclaimer

Research project only. Not a substitute for veterinary diagnosis.

## License

MIT (code). Datasets keep their original licenses.
