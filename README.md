# Supervised_cell_label_classification :vertical_traffic_light:

Author: Alexandros Papagiannakis.
HHMI at Stanford University 2021

This repository includes scripts used to detect and remove badly segmented cells.

If an SVM is used for training, the training function is included: train_the_SVM.py
The supervised prediction function is included: ml_prediction.py

A Jupyter notebook includes detailed directions for implementing the model training and supervised prediction: SVM_model_for_Unet_classification_for_GitHub.ipynb

Other models, such as a Naive-Bayes model, were also tested but not with as much prediction success as the weighted linear SVM.
