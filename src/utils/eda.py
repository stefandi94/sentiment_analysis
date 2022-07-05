import matplotlib.pyplot as plt
import pandas as pd

from consts import YELP_FULL_TRAIN

if __name__ == '__main__':
    df_full = pd.read_csv(YELP_FULL_TRAIN, header=None, names=['label', 'text'])

    print()