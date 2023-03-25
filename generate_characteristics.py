import tqdm
import argparse
from config import *
import pandas as pd
from characteristics.io.loader import TsvLoader
from characteristics.io.writer import TsvWriter
from characteristics.Dataset import GraphDataset

parser = argparse.ArgumentParser(description="Run generate characteristics.")
parser.add_argument('--dataset', type=str, default='bookcrossing')
parser.add_argument('--characteristics', type=str, nargs='+', default=ACCEPTED_CHARACTERISTICS)


def compute_characteristics_on_dataset(d_path, selected_characteristics):
    print(f'Dataset Path: {d_path}')
    # load dataset
    loader = TsvLoader(d_path + '/train.tsv')
    dataset = GraphDataset(loader.load())
    d_characteristics = {}
    iterator = tqdm.tqdm(selected_characteristics)
    for characteristic in iterator:
        iterator.set_description(f'Computing {characteristic}')
        d_characteristics.update({characteristic: dataset.get_metric(characteristic)})

    return d_characteristics


if __name__ == '__main__':

    args = parser.parse_args()

    # set dataset
    input_dataset = args.dataset
    dataset_folder = os.path.join(DATA_FOLDER, input_dataset)

    # set characteristics
    characteristics = args.characteristics
    characteristics = compute_characteristics_on_dataset(dataset_folder, characteristics)
    characteristics = {k: float(v) for k, v in characteristics.items()}

    # store results
    characteristics = pd.DataFrame(characteristics, index=[0])
    writer = TsvWriter(main_directory=OUTPUT_FOLDER, drop_header=False)
    writer.write(characteristics, file_name=f'characteristics', directory=input_dataset)
