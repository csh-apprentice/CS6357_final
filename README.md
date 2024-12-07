# Efficient MRI image segmentation using semi-supervised learning

This is the repository for our CS6357 final project. You can also get the paper in the report subfolder.

Key facts of our project:

- Fully supervised UNet3D baseline

- Semi-Supervised Method 1: Mean Teacher

- Semi-Supervised Method 2: Uncertainty Aware Mean Teacher Mode UAMT

## Before run

Get the datasets, we use the brats_itk_preprocessing.py in the /code/dataloaders folder for images prepossessing, but we have the prepossessed data in the google drive. https://drive.google.com/file/d/1U1cUpHjtc2-bjwhwRZB9mgziELvFWu2Z/view?usp=sharing and extract it in the data folder.

## How to run

**train:**

```
cd code
python train_XXXXX_3D.py 
```

**test**

```Linux
python test_XXXXX.py
```

where XXXXX refers to either fully_supervised or mean_teacher or uncertainty_aware_mean_teacher.

## Breakdown of the code structure

```
├── code                    <- Source code
│   ├── augmentations       <- Data augmentation scripts
│   │   ├── ctagument.py
│   │   └── __init__.py
│   ├── configs             <- Configuration scripts
│   ├── dataloaders         <- Data loader and preprocessing scripts
│   │   ├── brats_itk_preprocessing.py
│   │   ├── brats_preprocessing.py
│   │   ├── brats2019.py
│   │   ├── dataset.py
│   │   └── utils.py
│   ├── logs                <- Logging files
│   ├── networks            <- Model architecture scripts
│   │   ├── attention.py
│   │   ├── attention_unet.py
│   │   ├── discriminator.py
│   │   ├── efficient_encoder.py
│   │   ├── efficientunet.py
│   │   ├── encoder_tool.py
│   │   ├── enet.py
│   │   ├── grid_attention_layer.py
│   │   ├── net_factory.py
│   │   ├── net_factory_3d.py
│   │   ├── networks_other.py
│   │   ├── neural_network.py
│   │   ├── nnunet.py
│   │   ├── pnet.py
│   │   ├── swin_transformer_unet_skip_expand_decoder_sys.py
│   │   ├── unet_3D_dv_semi.py
│   │   ├── unet_3D.py
│   │   ├── utils.py
│   │   ├── vision_transformer.py
│   │   ├── vnet.py
│   │   └── VoxResNet.py
│   ├── utils               <- Utility scripts
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── ramps.py
│   │   └── util.py
│   ├── config.py           <- General configuration script
│   ├── convert.py          <- Data conversion scripts
│   ├── test_3D_util.py     <- Utility scripts for 3D testing
│   ├── train_fully_supervised_3D.py <- Script for fully supervised training
│   ├── train_mean_teacher_3D.py     <- Mean-teacher training script
│   ├── train_uncertainty_aware_mean_teacher_3D.py <- Uncertainty-aware training
│   ├── val_3D.py           <- Validation script
├── data                    <- Dataset and related files
│   ├── test.txt            <- Test dataset file
│   ├── train.txt           <- Train dataset file
│   └── README.md           <- Description of the data
├── environment.yml         <- Environment and dependencies file
├── README.md               <- Project description and documentation
├── .gitignore              <- Git ignore file
```

## Acknowledgement

We use the codebase from [HiLab-git/SSL4MIS: Semi Supervised Learning for Medical Image Segmentation, a collection of literature reviews and code implementations.](https://github.com/HiLab-git/SSL4MIS/tree/master)

as reference.
