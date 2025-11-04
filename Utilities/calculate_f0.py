import os
import pathlib
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt


f0_dict = {}
def calcualte_f0(file_path):
    y, sr = librosa.load(file_path)
    f0, voice_flag, voiced_probs = librosa.pyin(y, sr=sr, fmin=1,
                                                fmax=librosa.note_to_hz('C7'), frame_length=2048)
    f0 = np.nan_to_num(f0)
    
    min_f0 = round(np.min(f0), 2)
    max_f0 = round(np.max(f0), 2)
    mean_f0 = round(np.mean(f0), 2)

    return min_f0, max_f0, mean_f0


def load_file(folder_path):
    for count, filename in enumerate(os.listdir(folder_path)):
        if "wav" in filename:
            file_path = os.path.join(folder_path, filename)
            min_f0, max_f0, mean_f0 = calcualte_f0(file_path)
            f0_dict[filename] = str(min_f0)+","+str(max_f0)+","+str(mean_f0)

folder_path = 'DATASET_PATH'
load_file(folder_path)

