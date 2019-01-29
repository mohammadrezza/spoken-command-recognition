import os
import pickle


def collect_files(folder_path):
    files = []
    for file in os.listdir(folder_path):
        files.append((file.replace(".wav", ""), os.path.join(folder_path, file)))
    return files


def collect_folders(folder_path):
    folders = []
    for folder in os.listdir(folder_path):
        folders.append((folder, os.path.join(folder_path, folder)))
    return folders


def save(obj, fname, path):
    file = os.path.join(path, fname)
    with open(file, "wb") as f:
        f.write(pickle.dumps(obj))


def load(fname):
    with open(fname, "rb") as f:
        obj = pickle.loads(f.read())
    return obj
