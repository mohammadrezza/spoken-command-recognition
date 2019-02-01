import scipy.io.wavfile as wav
from utility import *
from proj_paths import *
import numpy as np
from HMM.extract_features import extract
import hmmlearn.hmm as hmm


# def hmm_init(states_num, samples_features):
def hmm_init(states_num):
    start_porb = np.ones((states_num))
    start_porb[0] = 0.9
    start_porb[1:states_num] *= 0.1 / (states_num - 1)

    trans_mat = np.zeros((states_num, states_num))

    for i in range(states_num):
        trans_mat[i][i] = 0.8
        if i + 1 < states_num:
            trans_mat[i][i + 1] = 0.2
        else:
            trans_mat[i][i] = 1

    # print(start_porb)
    # print(trans_mat)

    # states_points = [[] for _ in range(states_num)]
    #
    # for i in range(len(samples_features[0])):
    #     points = np.array_split(samples_features[i], states_num)
    #     for j in range(states_num):
    #         for p in points[j]:
    #             states_points[j].append(p)
    # cov = []
    # mean = []
    # for i in range(states_num):
    #     cov.append(np.cov(states_points[i]).tolist())
    #     mean.append(np.mean(states_points[i]).tolist())
    #
    # print(cov)
    # print(mean)

    model = hmm.GaussianHMM(n_components=states_num, startprob_prior=start_porb,
                            transmat_prior=trans_mat)

    # model = hmm.GaussianHMM(n_components=states_num)
    return model


if __name__ == "__main__":
    # np.seterr(all='raise')
    for folder_name, joined_folder in collect_folders(REF_VOICES_PATH):
        model = hmm_init(3)
        samples_feats = []
        length = []
        print("creating model for : " + folder_name)
        for voice_name, joined_voice in collect_files(joined_folder):
            _, sig = wav.read(joined_voice)
            feats = extract(sig)
            length.append(feats.shape[0])
            if samples_feats == []:
                samples_feats = feats
            else:
                samples_feats = np.concatenate((samples_feats, feats), axis=0)
        model.fit(samples_feats, lengths=length)
        save(model, folder_name, HMM_MODELS_PATH)
