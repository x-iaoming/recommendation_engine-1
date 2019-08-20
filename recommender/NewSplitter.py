from scipy.io import arff
from io import StringIO
import pandas as pd
import csv
import random
import numpy as np


class NewSplitter:
    """
    The NewSplitter class.

    Modified the Splitter Class in DRP to do test/train split for a csv data file
    """

    def __init__(self, num_splits=1, margin_percent=0.01, test_percent=0.33):
        """Specify the number of splits, naming pattern, and ratio of test/traning dataset sizes."""
        self.test_percent = test_percent
        self.margin_percent = margin_percent
        self.num_splits = num_splits

    def split(self, reactions, reactionsdf):
        """Actually perform the split."""
        test_index = []
        triplet_to_count, triplet_to_index, bad_index = self._count_compound_sets(
            reactions)
        key_counts = triplet_to_count.items()
        # set of tuples, where each tuple is a triplet that falls into the test corpus
        testkeys = self._get_test_keys(key_counts)
        #reactionsdf = reactions
        for i in range(len(reactionsdf)):
            if triplet_to_index[i] in testkeys:
                test_index.append(i)

        testdf = reactionsdf.iloc[test_index]
        traindf = reactionsdf.drop(test_index + bad_index, axis=0)

        testdf.to_csv("./sample_data/test.csv", index=False)
        traindf.to_csv("./sample_data/train.csv", index=False)

    def _get_test_keys(self, key_counts):
        random.shuffle(list(key_counts))
        total_size = sum(count for key, count in key_counts)
        print('Total size: {}'.format(total_size))
        goal_size = self.test_percent * total_size
        print('Goal size: {}'.format(goal_size))
        margin = total_size * self.margin_percent
        print('Margin: {}'.format(margin))
        test_size = 0

        # Determine which partitions should be tested.
        test_keys = set()
        for key, count in key_counts:
            if (abs(test_size + count - goal_size) < abs(test_size - goal_size)) and (test_size + count < goal_size + margin):
                # Adding the next set gets us closer to the goal size without
                # going over the error margin. Do it
                test_size += count
                test_keys.add(key)
            elif abs(test_size - goal_size) < margin:
                # We shouldn't add the next set and we're within the error
                # margin. We're done
                break
            # The next set is too big and we're not within the error margin.
            # Try skipping it
            else:
                raise RuntimeError(
                    'Failed to make a split under the given parameters.')

        return test_keys

    """
        Store the count of each unique triplet and skip those reactions that have missing value.
        Args:
            reactions:    a csv file that contains the reactions and result. 
    """

    def _count_compound_sets(self, reactions):
        with open(reactions) as csvfile:
            triplet_to_count = {}
            index_to_triplet = {}
            bad_index = []  # indices of rows that have missing values
            index = -1
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                index += 1
                InOrg1 = row[1]
                InOrg2 = row[4]
                Org1 = row[10]
                triplet = (InOrg1, InOrg2, Org1)
                index_to_triplet[index] = triplet
                if index > 0 and Org1 != '-1' and InOrg2 != '-1':
                    value = triplet_to_count.get(triplet, 0)
                    if value != 0:
                        triplet_to_count[triplet] += 1
                    else:
                        triplet_to_count[triplet] = 1
                else:
                    bad_index.append(index)
            return triplet_to_count, index_to_triplet, bad_index


# if __name__ == "__main__":
#     test = NewSplitter()

#     reactions = '/home/h205c/recommendation_engine/sample_data/nature17439-s2.csv'
#     a = test.split(reactions)
#     print(a)
