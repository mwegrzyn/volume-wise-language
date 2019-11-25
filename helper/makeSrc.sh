#!/bin/bash

# this script allows to transform notebooks into .py files
# which can then be run as a python module
# To achieve this, makeIpynbScripts uses jupyter nbconvert
# to turn ipynb's into py's, but omits all cells which have
# been tagged with "hide-cell", so all the interim outputs of
# the notebooks are omitted. 

echo "copy all models into scr"
cp -f ../models/* ../src/models

echo "copy external data into scr"
cp -f ../data/external/* ../src/data/external

echo "copy misc files needed for plotting"
cp -f ../reports/figures/legend.png ../src/visualization/
cp -f ../data/processed/csv/withinconclusive_prediction_df.csv ../src/models/

echo "convert scripts into py and put into src"
./makeIpynbScripts.sh ../notebooks/02-mw-make-z-maps.ipynb ../src/data/make_dataset_z_orig.py
./makeIpynbScripts.sh ../notebooks/03-mw-make-difference-ims.ipynb ../src/data/make_dataset_z_diff.py
./makeIpynbScripts.sh ../notebooks/09-mw-correlations-with-template.ipynb ../src/features/build_features.py
./makeIpynbScripts.sh ../notebooks/10-mw-train-test-classifier.ipynb ../src/models/predict_model.py
./makeIpynbScripts.sh ../notebooks/12-mw-make-correlation-plots-time.ipynb ../src/visualization/visualize_time.py
./makeIpynbScripts.sh ../notebooks/14-mw-prediction-space.ipynb ../src/visualization/visualize_counts.py
./makeIpynbScripts.sh ../notebooks/15-mw-visualize-logistic-regression.ipynb ../src/visualization/visualize_log.py
./makeIpynbScripts.sh ../notebooks/16-mw-individual-patients-plot.ipynb ../src/visualization/visualize_all.py
