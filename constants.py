import numpy as np

SAMPLE_RATE = 11025
FRAME_LENGTH = int(SAMPLE_RATE * 0.025)
FRAME_STEP = FRAME_LENGTH - int(SAMPLE_RATE * 0.015)
PRE_EMPH = 0.97
WINDOW_LENGTH = 0.025
WINDOW_STEP = 0.010
WINDOW_FUNCTION = np.hamming

DATASET_PATH = "dataset/"
MODELS_PATH = "models"
RECOGNIZE_PATH = "recognize/"

execute = {
    "alarm": "gnome-terminal",
    "google chrome": "google-chrome",
    "music": "spotify",
    "paint": "gimp",
    "telegram": "telegram-desktop"
}
