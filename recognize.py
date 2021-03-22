import scipy.io.wavfile as wav
from utils import *
from features import extract_features
from constants import *

if __name__ == "__main__":
    hmm_dict = dict()
    for raw_model_name, joined_model_path in collect_files(MODELS_PATH):
        model = load(joined_model_path)
        hmm_dict[raw_model_name] = model
    while True:
        try:
            for raw_file_name, joined_file_path in collect_files(RECOGNIZE_PATH):
                rate, sig = wav.read(joined_file_path)
                feat = extract_features(sig)
                best_score = -10_000_000
                cmd = None
                for name, model in hmm_dict.items():
                    score = model.score(feat)
                    print(f"result: {name} with score of {score}")
                    if int(score) > best_score:
                        best_score = score
                        cmd = execute[name]
                # uncomment line below to execute command
                # os.system(cmd)
                os.remove(joined_file_path)
        except Exception as e:
            print(e.__str__())
