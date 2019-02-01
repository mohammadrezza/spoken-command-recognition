import scipy.io.wavfile as wav
from utility import *
from proj_paths import *
from HMM.extract_features import extract

hmm_dict = dict()
execute = dict()
execute["alarm"] = "gnome-terminal"
execute["google chrome"] = "google-chrome"
execute["music"] = "spotify"
execute["paint"] = "gimp"
execute["telegram"] = "telegram-desktop"


def keep_predicting():
    # remove previous files
    while True:
        try:
            for raw_file_name, joined_file_path in collect_files(REAL_TIME_PATH):
                rate, sig = wav.read(joined_file_path)
                feat = extract(sig)
                best_score = -10_000_000
                cmd = None
                for name, model in hmm_dict.items():
                    score = model.score(feat)
                    print(name, score)
                    if int(score) > best_score:
                        best_score = score
                        cmd = execute[name]
                os.system(cmd)
                os.remove(joined_file_path)
        except Exception as e:
            # print(e.__str__())
            pass


if __name__ == "__main__":
    for raw_model_name, joined_model_path in collect_files(HMM_MODELS_PATH):
        model = load(joined_model_path)
        hmm_dict[raw_model_name] = model
    keep_predicting()
