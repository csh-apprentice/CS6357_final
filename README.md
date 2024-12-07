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

```



## Acknowledgement

We use the codebase from [HiLab-git/SSL4MIS: Semi Supervised Learning for Medical Image Segmentation, a collection of literature reviews and code implementations.](https://github.com/HiLab-git/SSL4MIS/tree/master)

as reference.
