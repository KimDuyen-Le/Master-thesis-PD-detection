# Template for Speaker Identification Modified to suit Binary Classification for Detecting PD

This folder provides the necessary files for training a binary classification model from scratch based on the tutorial 
"Speech Classification From Scratch" (https://speechbrain.readthedocs.io/en/latest/tutorials/tasks/speech-classification-from-scratch.html).
The PC-GITA database is used. More details about what each file does and the modifications made are within each file.

There are four files here:
* `train.py`: the main code file, outlines the entire training process.
* `train.yaml`: the hyperparameters file, sets all parameters of execution.
* `custom_model.py`: A file containing the definition of a PyTorch module.
* `mini_librispeech_prepare.py`: If necessary, downloads and prepares data manifests.

To train the binary classification model, execute the following on the command-line:

```bash
python train.py train.yaml
```
