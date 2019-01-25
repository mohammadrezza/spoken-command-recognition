import scipy.io.wavfile as wav
from utility import *
from proj_paths import *
from DTW.extract_features import extract

if __name__ == "__main__":
    for voice in collect_files(REF_VOICES):
        _, sig = wav.read(voice)
        feats = extract(sig)
