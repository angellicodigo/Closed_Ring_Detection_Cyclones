# Detection of Nearly Closed Surface Wind Field of Medicanes

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Medicanes are special kind of cyclones that appear only in the Mediterranean Sea. One of their properties described by The Institute of Atmospheric Sciences and Climate (CNR-ISAC) is having a strong near-surface wind field with a nearly closed cyclonic structure. This project aims to use machine learning to detect these closed ring of maximum wind of cyclones, including Medicanes. We are using Advanced Scatterometer (ASCAT) wind data at 12.5 km resolution.

This project is exploring the use of object detection (Faster R-CNN) and semantic segmentaiton (U-Nets) models. The dataset is not uploaded on the repository to save storage.

I also included a presentation that includes background information and more details of the project.

# Directory structure (TODO)

# How to run the code?
1. Create the environment
```
$ cd /Nearly_Closed_Ring_Detection_of_Cyclones/src
$ conda env create -f environment.yml
```
2. Fill out the paths in .env file
3. (Optional) To create the dataset, run these commands
```
cd src/data/script
python generate_dataset.py
python preprocess_dataset.py
python filter_dataset.py --radius --threshold --n
```
4. Train the model
```
cd src
python train.py
```
