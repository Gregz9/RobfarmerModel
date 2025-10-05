from utilities import DataHandler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dset", type=str, default="Robofarmer-II")

dataset_annots = {
    # "Robofarmer": "../data/Annotation.csv",
    "Robofarmer-II": "../data/datasets/Robofarmer-II/Annotation.csv",
}

if __name__ == "__main__":

    args = parser.parse_args()

    dataHandler = DataHandler(dataset_annots[args.dset], args.dset)
    # dataHandler.dumpJsonAnnotation()
