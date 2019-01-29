import numpy as np
from utility import *
from proj_paths import *


hmm_dict = dict()

if __name__ == "__main__":
    hmm_dict = dict()
    for raw_model_name, joined_model_path in collect_files(HMM_MODELS_PATH):
        model = load(joined_model_path)
        hmm_dict[raw_model_name] = model
