import scipy.io.wavfile as wav
import hmmlearn.hmm as hmm
from features import extract_features
from utils import *
from constants import *


def hmm_init(states_num):
    start_porb = np.ones(states_num)
    start_porb[0] = 0.9
    start_porb[1:states_num] *= 0.1 / (states_num - 1)

    trans_mat = np.zeros((states_num, states_num))

    for i in range(states_num):
        trans_mat[i][i] = 0.8
        if i + 1 < states_num:
            trans_mat[i][i + 1] = 0.2
        else:
            trans_mat[i][i] = 1

    return hmm.GaussianHMM(n_components=states_num, startprob_prior=start_porb, transmat_prior=trans_mat)


if __name__ == "__main__":
    for folder_name, joined_folder in collect_folders(DATASET_PATH):
        model = hmm_init(3)
        samples_feats = []
        length = []
        print("creating model for : " + folder_name)
        for voice_name, joined_voice in collect_files(joined_folder):
            _, sig = wav.read(joined_voice)
            feats = extract_features(sig)
            length.append(feats.shape[0])
            if samples_feats == []:
                samples_feats = feats
            else:
                samples_feats = np.concatenate((samples_feats, feats), axis=0)
        model.fit(samples_feats, lengths=length)
        save(model, folder_name, MODELS_PATH)
