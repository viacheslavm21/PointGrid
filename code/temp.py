import glob


def get_file_name(file_path):
    parts = file_path.split('/')
    part = parts[-1]
    parts = part.split('.')
    return parts[0]

TRAINING_FILE_LIST = [get_file_name(file_name) for file_name in glob.glob('../data/ShapeNet/train/' + '*.mat')]

print(TRAINING_FILE_LIST)
