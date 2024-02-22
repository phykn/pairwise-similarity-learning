import yaml


def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def fix_dtype(data):
    data['image'] = data['image'].float()
    data['label'] = data['label'].long()
    return data

def load_yaml(path):
    with open(path, mode = 'r') as f:
        obj = yaml.load(f, Loader = yaml.FullLoader)
    return obj