from mutual_information import MutualInformation
import pandas as pd

training_set = pd.read_csv('nature17439-s2.csv', low_memory=False)
training_set = training_set.replace('?', 0)
test_set = pd.read_csv('nature17439-s3.csv')

# Filtering XXX columns
descriptors_to_keep_common = [col for col in training_set if 'XXX' not in col]

descriptors_to_remove_training = ['outcome']

descriptors_to_keep_training = [
    col for col in descriptors_to_keep_common if col not in descriptors_to_remove_training]

descriptors_to_remove_testing = ['outcome (actual)', 'predicted outcome']

mi = MutualInformation(training_set, 'outcome', descriptors_to_keep_training)

testing_descriptors = test_set[descriptors_to_keep_training].values


print([mi.get_delta_mutual_info(desc) for desc in testing_descriptors])
