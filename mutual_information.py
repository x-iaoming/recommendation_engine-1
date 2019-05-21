import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import multivariate_normal
from typing import Any


DESCRIPTORS_TO_REMOVE = ['compound_0_amount_grams', 'compound_1_amount_grams',
                         'compound_2_amount_grams', 'labGroup', 'notes', 'compound_0_role',
                         'compound_1_role', 'compound_2_role', 'compound_0', 'compound_1',
                         'compound_2', 'compound_0_amount', 'compound_1_amount',
                         'compound_2_amount']


class Pdf:
    def __init__(self, group: dict) -> None:
        self.group = group
        self.success_count = self.group['result'].count(True)
        self.descriptor_normal = self.calc_pdf()
        self.success_descriptor_normal, self.failure_descriptor_normal = \
            self.calc_joint_pdfs()

    def calc_pdf(self) -> None:
        """
        Probability Density Function of a particular reaction
        Args:
            group: Single reaction dictionary. Key = set of chemicals
                   value = descriptors
        Returns:
            descriptors_normal: pdf of reaction

        """

        if self.success_count <= 1 or self.success_count >= len(group['result']) - 1:
            return None
        else:
            self.success_mean = self.success_count/float(len(group['result']))
            self.fail_mean = 1 - self.success_mean

            self.descriptor_mean = np.mean(group['descriptors'], axis=0)
            self.descriptor_cov = np.cov(group['descriptors'].T)

            try:
                descriptor_normal = multivariate_normal(
                    mean=self.descriptor_mean, cov=self.descriptor_cov, allow_singular=True)
            except Exception as e:
                raise e

        return descriptor_normal

    def calc_joint_pdfs(self):
        if self.success_count <= 1 or self.success_count >= len(group['result']) - 1:
            return (None, None)
        else:
            success_indices = [idx for idx, val in
                               enumerate(group['result']) if val == True]
            success_descriptors = np.array([group['descriptors'][i]
                                            for i in success_indices])
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


class MutualInformation:
    def __init__(self, dataset: pd.DataFrame) -> None:
        """
        Mutual information calculator

        Args:
            dataset: Full dataset over which to calculate MI
        """
        self.result_column_label = 'boolean_crystallisation_outcome_manual_0'
        self.dataset = dataset  # type:pd.DataFrame
        self.descriptors_to_keep = [
            col for col in dataset.columns if col not in DESCRIPTORS_TO_REMOVE]
        self.descriptors = dataset[self.descriptors_to_keep]
        self.labels = dataset[self.result_column_label]

        self.groups = self.group_dataset(self.dataset)

        self.groups = self.generate_pdfs(self.groups)
        print(len(self.dataset))
        print(len(self.groups))

    def group_dataset(self, dataset: pd.DataFrame) -> defaultdict:
        """
        Creates a dictionary by grouping reactions with the same chemicals

        Args: 
            dataset: Pandas Dataframe with all data
        Returns:
            defaultdict: Dictionary of groups of chemicals and their descriptors
        """
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
        """
        Generates pdfs for a particular reaction
        and joint pdfs for their reaction success or failure
        Args:
            groups: Dictionary of reactions
        Returns:
            groups: Dictionary of reactions with pdfs and joint_pdfs
        """
        for key in groups:
            groups[key]['pdf'] = Pdf(groups[key])
        return groups

    def calculate_mutual_information(self, groups: defaultdict) -> defaultdict:
        for key, value in groups.items():
            groups['MI'] = None


if __name__ == "__main__":
    # dataset = pd.read_csv('~/test_data.csv')
    dataset = pd.read_csv('~/test_data_full.csv', low_memory=False)
    dataset_org = dataset.iloc[:-100]
    dataset_new = dataset.iloc[-100:]
    mi = MutualInformation(dataset_org)
