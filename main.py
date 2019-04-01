import pandas as pd
import more_itertools as mit
from random import shuffle
from TabuSearch import TabuSearch
import sys


def check_nan(data, cols):
    # whether there are nan in the selected columns
    return data[cols].isnull().any().any()

def main():
    filename = sys.argv[1]
    group = int(sys.argv[2])
    cols = sys.argv[3].split(',')
    data = pd.read_csv(filename)
    if check_nan(data, cols):
        raise ValueError('There are NAN values in the selected columns')
    else:
        dummy_cols = [list(data[col].dropna().unique()) for col in cols]
        for col in cols:
            data = pd.concat([data, pd.get_dummies(data[col])], axis = 1)
        order = [i for i in range(0, data.shape[0])]
        shuffle(order)
        initial = [list(x) for x in mit.divide(group, order)]
        algorithm = TabuSearch(data, dummy_cols, initial, 10, 500, max_score=None)
        results = algorithm.run()
        result_filename = filename.split('.')[0] + '_group.csv'
        results.to_csv(result_filename, index=False)
        print('Grouping done!')

if __name__ == "__main__":
    main()