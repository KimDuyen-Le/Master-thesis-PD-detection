"""
Downloads and creates data manifest files for PC-GITA.
The manifest files are created from the dataset that 
is already split into training, validation and test set.

"""

import json
import os
import random
import shutil
import pandas
import librosa

from speechbrain.dataio.dataio import read_audio
from speechbrain.utils.data_utils import download_file, get_all_files
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)
MINILIBRI_TRAIN_URL = "http://www.openslr.org/resources/31/train-clean-5.tar.gz"
SAMPLERATE = 16000
    
def prepare_mini_librispeech(
    data_folder,
    train_csv,
    valid_csv,
    test_csv,
    save_json_train,
    save_json_valid,
    save_json_test,
):
    """
    Prepares the json files for the PC-GITA dataset.

    Arguments
    ---------
    fold: int
        Fold number to process (1-10)
    train_csv : str
        Path to the train CSV file.
    valid_csv : str
        Path to the valid CSV file.
   test_csv : str
        Path to the test CSV file.
    save_json_test : str
        Path where the test data specification file will be saved.
    save_json_valid : str
        Path where the valid data specification file will be saved.
    save_json_train : str
        Path where the train data specification file will be saved.    
    Returns
    -------
    None
    """
    
    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return
    
    # Creating json files
    create_json(train_csv, save_json_train)
    create_json(valid_csv, save_json_valid)
    create_json(test_csv, save_json_test)
    
def get_audio_duration(audio_file):
    
    # Load the audio files to obtain audio file duration
    audio, sr = librosa.load(audio_file, sr=None)
    duration = librosa.get_duration(y=audio, sr=sr)
    return duration

def create_json(csv_file, json_file, data_root="{data_root}"):
    """
    Creates the json file from CSV files with meta data.

    Arguments
    ---------
    csv_file:  str
        The path to the CSV files.
    json_file : str
        The path of the output json file
    data_root : str
        Root path placeholder for audio file paths in the JSON.
    """
    
    # Load the CSV file into a pandas DataFrame
    df = pandas.read_csv(csv_file)

    # Initialize dicitonary to store JSON data
    json_dict = {}
    
    # Iterate through rows of DataFrame
    for _, row in df.iterrows():
        audio_file = row['audio_path']  # Audio file path
        uttid = row['id']  # Unique utterance ID
        spk_id = row['status'].lower()  # 'pd' or 'hc'

        # Reading the signal (to get duration in seconds)
        duration = get_audio_duration(audio_file)

        # Create entry for this utterance
        json_dict[uttid] = {
            "wav": os.path.join(data_root, audio_file),
            "length": duration,
            "spk_id": spk_id,
            }

    # Writing the dictionary to the json file
    with open(json_file, mode="w", encoding="utf-8") as json_f:
        json.dump(json_dict, json_f, indent=2)
    logger.info(f"{json_file} successfully created!")


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Arguments
    ---------
    *filenames: tuple
        The path to files that should exist in order to consider
        preparation already completed.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True
