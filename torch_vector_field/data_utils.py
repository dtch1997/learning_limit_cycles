import numpy as np

def load_dataset(dataset_name: str):
    xs = np.load(f'data/{dataset_name}_x.npy')
    dot_xs = np.load(f'data/{dataset_name}_dx.npy')
    return xs, dot_xs

def save_dataset(xs, dot_xs, dataset_name: str):
    np.save(f'data/{dataset_name}_x.npy', xs)
    np.save(f'data/{dataset_name}_dx.npy', dot_xs)