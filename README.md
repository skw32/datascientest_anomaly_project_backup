Project : Anomaly Detection in Industrial Products
==============================

In this project, we detected anomalies in industrial product images using machine learning and deep learning models. We also analyzed the product images and performed data visualization and modeling to better understand and interpret the results.

The dataset used in this study is the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset. MVTec AD is a benchmark dataset for evaluating anomaly detection methods, with a focus on industrial inspection. It contains over 5,000 high-resolution images divided into fifteen object and texture categories. Each category includes a set of defect-free training images and a test set containing images with various types of defects as well as defect-free images.

We have done the following steps in our project:
1. Data Exploration & Visualization: Analyze product images to understand and identify defect patterns.
2. Image Preprocessing & Feature Engineering: Prepare and transform images for ML/DL models.
3. Machine Learning & Deep Learning Models: Design and train models such as Random Forest, CNN and Transfer Learning models.
4. Model Evaluation: Assess performance of trained models.
5. Prediction Interpretation: Use Grad-CAM to visualize defect regions.

Read our report: [Final Report](https://github.com/kavithaAra/nov25_cds_anomalie-backup/blob/main/reports/Rendu%203%20rapport%20final.pdf)

Explore the anomaly detection demo: [Industrial Anomaly Detection App](https://industrial-anomaly-detection.streamlit.app/)

This repo contains jupyter notebooks, reports and final models for our project.

To create python environments for running different components of the project, different requirements.txt files will be provided:
- requirements.txt: for running the streamlit application for demos of the final models
- requirements\_skw32.txt: for running any notebooks by author skw32
- requirementsi\_Karine\_KAVITHA.txt: for running any notebooks by author kavithaAra
- Requirementes\_makhlouf\_hanouti: for running any notebooks by author COMHANOUTI
 

 * Les modèles avec segmentation entraînés realisés par Makhlouf HANOUTI étant volumineux, ils ne sont pas stockés dans le dépôt GitHub. Ils sont disponibles via le lien Google Drive suivant: https://drive.google.com/drive/folders/1-NRbQwA5-CATleMWONc361DOFB_-7Qcs?usp=drive_link.


Project Organization
------------


  

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
