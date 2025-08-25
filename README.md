# Metal Sheet Defects
==============================

## A model to classify the defects in steel sheets and identify their location

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    ├── gui                <- Created app to display the model results on an input image
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
    
    
Setting up the virtual environment
------------
1. Install the modules specified in requirements.txt
2. Clone this repository, add it to your drive if you wish to use google colab
3. If the data is stored on your drive you can use google colab to run the given command line 
4. In your google colab notebook run this command to mount it on your drive.  
 ```
 from google.colab import drive
 drive.mount("/content/drive")
 ```

Running the project 
------------
1. Set up the virtual environment
2. run the given command as per the specified command line arguments
```
  argv[0]=name of the script file
  argv[1]=training (boolean variable for training or not)
  argv[2]=testing (boolean variable for testing or not)
  argv[3]=Resume (boolean variable for resuming) 
  argv[4]=device (with or without GPU) 
  argv[5]=dataset_path (path to the dataset, the common directory where training data and testing data is stored)
  argv[6]=num epochs
  argv[7]=classification model (select one of multiple models) 
  argv[8]=input size of images to model (height)
  argv[9]=input size of images to model (width)
  argv[10]=batch size 
  argv[11]=learning rate
  
  !python3 train_model.py argv[1] argv[2] argv[3] argv[4] argv[5] argv[5] argv[6] argv[7] argv[8] argv[9] argv[10] argv[11]
 ```
for example, it would look like this on your local system:
```
!python3 '/content/drive/My Drive/metal-sheet-defect/src/models/train_model.py'  0 0 1 0 '/content/drive/My Drive/Datasets/caliche/severstal-steel-defect-detection' 35 1 256 1600 8 0.001 
```
Running the app
------------
1. Set up the virtual environment and run app.py from the gui directory as follows
2. Update the path of the hdf5 file as per your system in app.py
3. run the app using the following command 
 ```
 streamlit run app.py
 ```

## Sample Outputs
Once uploaded, the GUI displays the defects as follows:

<img width="890" height="181" alt="defect_multi_1" src="https://github.com/mahekvora/Metal-Sheet-Defects-Pixelwise-Classification/blob/master/reports/figures/defect_multi_1.png" />

<img width="890" height="181" alt="defect_4_1" src="https://github.com/mahekvora/Metal-Sheet-Defects-Pixelwise-Classification/blob/master/reports/figures/defect_4_1.png" />

--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
