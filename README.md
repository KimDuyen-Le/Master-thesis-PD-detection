# Automatic Speech Analysis for Detection of Parkinson’s Disease using X-Vector Speech Representations

This is for the course TFE4940 Electronic Systems Design and Innovation, Master's Thesis at NTNU, by Duyen Kim Le.

This git repository contains code for the thesis which explores an automatic detection of PD based on speech analysis by utilizing x-vector speech representations from both a pre-trained and custom-trained x-vector model from SpeechBrain.

# About the files
* code:
This folder contains the main code file for the binary classification system with x-vector embeddings from x-vector models and classification models.
 
* metadata:
This folder contains a file with the meta data about the PD patients and HC speakers in the PC-GITA database.

* customtrained-xvector-model:
This folder provides the necessary files for training a binary classification model from scratch based on the tutorial 
"Speech Classification From Scratch" (https://speechbrain.readthedocs.io/en/latest/tutorials/tasks/speech-classification-from-scratch.html).

There are four files here:
* `train.py`: the main code file, outlines the entire training process.
* `train.yaml`: the hyperparameters file, sets all parameters of execution.
* `custom_model.py`: A file containing the definition of a PyTorch module.
* `mini_librispeech_prepare.py`: If necessary, downloads and prepares data manifests.
