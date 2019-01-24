import os


def collect_files(folder_path):
    files = []
    for file in os.listdir(folder_path):
        files.append(os.path.join(folder_path, file))
    return files
