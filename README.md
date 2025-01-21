# Automatic Speech Analysis for Detection of Parkinson’s Disease using X-Vector Speech Representations

This is for the course TFE4940 Electronic Systems Design and Innovation, Master's Thesis at NTNU, by Duyen Kim Le.

This git repository contains code for the thesis which explores an automatic detection of PD based on speech analysis by utilizing x-vector speech representations from both a pre-trained and custom-trained x-vector model from SpeechBrain.

# About the files

# Template for Speaker Identification Modified to suit Binary Classification for Detecting PD

This folder provides the necessary files for training a binary classification model from scratch based on the tutorial 
"Speech Classification From Scratch" (https://speechbrain.readthedocs.io/en/latest/tutorials/tasks/speech-classification-from-scratch.html).
The PC-GITA database is used. More details about what each file does and the modifications made are within each file.

There are four files here:
* `train.py`: the main code file, outlines the entire training process.
* `train.yaml`: the hyperparameters file, sets all parameters of execution.
* `custom_model.py`: A file containing the definition of a PyTorch module.
* `mini_librispeech_prepare.py`: If necessary, downloads and prepares data manifests.

To train the binary classification model, the following on the command-line is executed:

```bash
python train.py train.yaml
```
