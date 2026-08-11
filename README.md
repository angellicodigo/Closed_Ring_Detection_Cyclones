# Detection of Nearly Closed Surface Wind Field of Medicanes

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Medicanes are special kind of cyclones that appear only in the Mediterranean Sea. One of their properties described by The Institute of Atmospheric Sciences and Climate (CNR-ISAC) is having a strong near-surface wind field with a nearly closed cyclonic structure. This project aims to use machine learning to detect these closed ring of maximum wind of cyclones, including Medicanes. We are using Advanced Scatterometer (ASCAT) wind data at 12.5 km resolution.

This project is exploring the use of object detection (Faster R-CNN) and semantic segmentaiton (U-Nets) models. The dataset is not uploaded on the repository to save storage.

I also included a presentation that includes background information and more details of the project.

# Directory structure (Based on the Cookiecutter Data Science):
```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         {{ cookiecutter.module_name }} and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             
    │
    ├── config                  <- Store useful variables and configuration
    │   ├── __init__.py 
    │   ├── loss.py             <- Code that stores the loss functions used         
    │   ├── models.py           <- Code that stores the structure of the model used
    │   ├── partialconv2d.py    <- Code of the Partial Convolution Layer fron Nvidia
    │   └── utils.py            <- Code of functions that may be reused throughout other files
    │
    ├── dataset                 <- Scripts to download or generate data
    │   ├── __init__.py    
    │   ├── generate_dataset.py      <- Code to generate annotations_raw.txt from annotations_template.txt
    │   ├── preprocess_dataset.py    <- Code to create annotations_interm.txt from annotations_raw.txt
    │   ├── filter_dataset.py        <- Code where given number of points over the ocean and percentage of points
    │   │                               over the ocean over all points, filter annotations_interm.txt to get annotations.txt
    │   ├── visualize_dataset.py     <- Code to visualize any version of annotations.txt   
    │   └── dataset.py               <- Code of a custom Pytorch Dataloader that loads annotations.txt
    │
    ├── plot                    <- Scripts to create visualizations
    │   ├── plot_stats.py       <- Code to create box plots, histograms, scatterplots, or density plots. Also
    │   │                          identifies outliers and the ideal parameters for filter_dataset.py.        
    │   └── plot.py             <- Code to plot either each data point in annotations.txt or a single ASCAT file
    │
    └── train.py                <- Code to train the model 
```

# How to run the code?
1. Create the environment
```bash
$ cd /Nearly_Closed_Ring_Detection_of_Cyclones/src
$ conda env create -f environment.yml
```
