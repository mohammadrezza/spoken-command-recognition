from python_speech_features import sigproc, mfcc, delta
from constants import *


def zcr(frames):
    def sign(x):
        return 1 if x >= 0 else -1

    zcrs = []
    for frame in frames:
        zc_rate = 0
        for i in range(1, len(frame)):
            zc_rate += abs(sign(frame[i]) - sign(frame[i - 1])) / 2
        zcrs.append(zc_rate / len(frame))
    return zcrs


def concat_feats(frames_feats, feat_coeffs):
    return np.concatenate((frames_feats, feat_coeffs), axis=1)


def auto_correlate(frames, eta):
    energies = []
    for frame in frames:
        total_sum = 0
        for i in range(eta, len(frame)):
            total_sum += frame[i] * frame[i - eta]
        energy = 1 / len(frame) * total_sum
        energies.append(energy)
    return energies


def extract_features(sig):
    # framing
    sig_frames = sigproc.framesig(sig=sig, frame_len=FRAME_LENGTH, frame_step=FRAME_STEP)

    # calculate mfcc features
    mfcc_feat = mfcc(signal=sig, samplerate=SAMPLE_RATE, winlen=WINDOW_LENGTH, winstep=WINDOW_STEP,
                     numcep=13, preemph=PRE_EMPH, winfunc=WINDOW_FUNCTION)
    mfcc_feat_delta = delta(mfcc_feat, 20)
    mfcc_feat_delta_delta = delta(mfcc_feat_delta, 20)

    # calculate zero cross rating
    zcrs = zcr(sig_frames)
    zcrs = np.array([zcrs]).reshape(len(zcrs), 1)

    # calculate energy
    energies = auto_correlate(sig_frames, 0)
    energies = np.array([energies]).reshape(len(energies), 1)

    frames_feats = mfcc_feat
    frames_feats = concat_feats(frames_feats, mfcc_feat_delta)
    frames_feats = concat_feats(frames_feats, mfcc_feat_delta_delta)
    frames_feats = concat_feats(frames_feats, zcrs)
    frames_feats = concat_feats(frames_feats, energies)

    return frames_feats
