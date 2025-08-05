import pickle
import pandas as pd
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--path', '-p', type=str, required=True, help='Path to the folder containing the results')
args = argparser.parse_args()

if __name__ == '__main__':
    with open(args.path, 'rb') as f:
        pickle_file = pickle.load(f)

    pick
    print(len(pickle_file))