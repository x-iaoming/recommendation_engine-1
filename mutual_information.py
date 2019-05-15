import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import multivariate_normal
from typing import Any


DESCRIPTORS_TO_REMOVE = ['compound_0_amount_grams', 'compound_1_amount_grams', 'compound_2_amount_grams', 'labGroup', 'notes', 'compound_0_role',
                         'compound_1_role', 'compound_2_role', 'compound_0', 'compound_1', 'compound_2', 'compound_0_amount', 'compound_1_amount',
                         'compound_2_amount']


class MutualInformation:
    def __init__(self, dataset: pd.DataFrame) -> None:
        self.result_column_label = 'boolean_crystallisation_outcome_manual_0'
        self.dataset = dataset
        self.descriptors_to_keep = [
            col for col in dataset.columns if col not in DESCRIPTORS_TO_REMOVE]
        self.descriptors = dataset[self.descriptors_to_keep]
        self.labels = dataset[self.result_column_label]

        self.groups = self.group_dataset(self.dataset)

        self.groups = self.generate_pdfs(self.groups)

    def group_dataset(self, dataset: pd.DataFrame) -> defaultdict:
        groups = defaultdict(dict)  # type:defaultdict
        for idx, row in self.dataset.iterrows():
            key = frozenset((row['compound_0'],
                             row['compound_1'], row['compound_2']))
            grp = defaultdict(list)  # type:defaultdict
            grp['descriptors'].append(self.descriptors.iloc[idx].values)
            grp['descriptors'] = np.array(grp['descriptors'])
            grp['result'].append(self.labels.iloc[idx])
            groups[key] = grp

        return groups

    def generate_pdfs(self, groups: defaultdict) -> defaultdict:
        for key in groups:
            groups[key]['pdf'] = self.pdf(groups[key])
            groups[key]['joint_pdf'] = self.joint_pdf(groups[key])

        return groups

    def pdf(self, group: defaultdict) -> Any:
        success_count = group['result'].count(True)

        if success_count <= 1 or success_count >= len(group['result']) - 1:
            return None
        else:
            success_mean = success_count/float(len(group['result']))
            fail_mean = 1 - success_mean

            descriptor_mean = np.mean(group['descriptors'], axis=0)
            descriptor_cov = np.cov(group['descriptors'].T)

            try:
                descriptor_normal = multivariate_normal(
                    mean=descriptor_mean, cov=descriptor_cov, allow_singular=True)
            except Exception as e:
                raise e

        return descriptor_normal

    def joint_pdf(self, group: defaultdict) -> tuple:
        success_count = group['result'].count(True)

        if success_count <= 1 or success_count >= len(group['result']) - 1:
            return (None, None)
        else:
            success_indices = [idx for idx, val in
                               enumerate(group['result']) if val == True]
            success_descriptors = np.array([group['descriptors'][i]
                                            for i in success_indices])
            print(success_indices)

            try:
                success_descriptor_normal = multivariate_normal(mean=np.mean(success_descriptors, axis=0),
                                                                cov=np.cov(
                                                                    success_descriptors.T),
                                                                allow_singular=True)
            except Exception as e:
                raise e

            failure_descriptors = np.array(
                [desc for desc in group['descriptors'] if desc not in success_descriptors])

            try:
                failure_descriptor_normal = multivariate_normal(mean=np.mean(failure_descriptors, axis=0),
                                                                cov=np.cov(
                                                                    failure_descriptors.T),
                                                                allow_singular=True)
            except Exception as e:
                raise e

        return (success_descriptor_normal, failure_descriptor_normal)


if __name__ == "__main__":

    dataset = pd.read_csv('~/test_data.csv')
    mi = MutualInformation(dataset)
