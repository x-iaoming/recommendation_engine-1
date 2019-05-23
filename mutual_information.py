import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import multivariate_normal
import math
import copy
from typing import Any


DESCRIPTORS_TO_REMOVE = ['compound_0_amount_grams', 'compound_1_amount_grams',
                         'compound_2_amount_grams', 'labGroup', 'notes', 'compound_0_role',
                         'compound_1_role', 'compound_2_role', 'compound_0', 'compound_1',
                         'compound_2', 'compound_0_amount', 'compound_1_amount',
                         'compound_2_amount']

SUCCESS = True
FAILURE = False


class Pdf:
    def __init__(self, group: dict) -> None:
        self.group = group
        self.success_count = self.group['result'].count(True)
        #self.success_count = self.count_successes()
        self.descriptor_normal = self.calc_pdf()
        self.success_descriptor_normal, self.failure_descriptor_normal = \
            self.calc_joint_pdfs()
        self.mutual_information = self.calculate_mutual_information()

    def count_successes(self):
        success = 0
        success += self.group['result'].count(3)
        success += self.group['result'].count(4)
        return success

    def calc_pdf(self) -> Any:
        """
        Probability Density Function of a particular reaction
        Args:
            group: Single reaction dictionary. Key = set of chemicals
                   value = descriptors
        Returns:
            descriptors_normal: pdf of reaction

        """
        descriptor_normal = None
        if self.success_count <= 1 or len(self.group['result']) - self.success_count <= 1:
            return None
        else:
            self.success_mean = self.success_count / \
                float(len(self.group['result']))
            self.fail_mean = 1 - self.success_mean

            self.descriptor_mean = np.mean(
                self.group['descriptors'], axis=0)
            self.descriptor_cov = np.cov(self.group['descriptors'].T)

            try:
                descriptor_normal = multivariate_normal(
                    mean=self.descriptor_mean, cov=self.descriptor_cov, allow_singular=True)
            except Exception as e:
                raise e
        return descriptor_normal

    def calc_joint_pdfs(self):
        if self.success_count <= 1 or len(self.group['result']) - self.success_count <= 1:
            return (None, None)
        else:
            success_indices = [idx for idx, val in
                               enumerate(self.group['result']) if val == True]
            success_descriptors = np.array([self.group['descriptors'][i]
                                            for i in success_indices])
            try:
                success_descriptor_normal = multivariate_normal(mean=np.mean(success_descriptors, axis=0),
                                                                cov=np.cov(
                                                                    success_descriptors.T),
                                                                allow_singular=True)
            except Exception as e:
                raise e

            failure_indices = [idx for idx, val in
                               enumerate(self.group['result']) if val == False]
            failure_descriptors = np.array([self.group['descriptors'][i]
                                            for i in failure_indices])
            try:
                failure_descriptor_normal = multivariate_normal(mean=np.mean(failure_descriptors, axis=0),
                                                                cov=np.cov(
                                                                    failure_descriptors.T),
                                                                allow_singular=True)

            except Exception as e:
                raise e

        return (success_descriptor_normal, failure_descriptor_normal)

    def calculate_mutual_information(self):
        mutual_information = None
        if self.descriptor_normal:
            mutual_information = 0.0
            for row in self.group['descriptors']:
                mutual_information += self.row_mutual_information(row)
        return mutual_information

    def row_mutual_information(self, row: list) -> float:
        mutual_information = 0.0
        feature = self.prob_feature(row)
        if feature:
            joint = self.prob_joint(row, SUCCESS)
            if joint != 0:
                mutual_information += joint * \
                    math.log(joint / (self.success_mean*feature), 2)
            joint = self.prob_joint(row, FAILURE)
            if joint > 1e-300:
                print(joint, self.fail_mean * feature)
                mutual_information += joint * \
                    math.log(joint / (self.fail_mean*feature), 2)
        return mutual_information

    def prob_feature(self, row):
        if self.descriptor_normal:
            return self.descriptor_normal.pdf(row)
        else:
            return 0.0

    def prob_joint(self, row, result):
        if result:
            return self.success_descriptor_normal.pdf(row)
        else:
            return self.failure_descriptor_normal.pdf(row)

    def delta_mi(self, candidate):
        new_group = copy.copy(self.group)
        new_group['descriptors'] = np.vstack(
            (new_group['descriptors'], candidate))
        new_pdf = Pdf(new_group)
        new_mi = new_pdf.mutual_information
        return abs(self.mutual_information - new_mi)


class MutualInformation:
    def __init__(self, dataset: pd.DataFrame, result_column_label: str, descriptors_to_keep: list) -> None:
        """
        Mutual information calculator

        Args:
            dataset: Full dataset over which to calculate MI
        """
        self.result_column_label = result_column_label
        self.dataset = dataset  # type:pd.DataFrame
        self.descriptors_to_keep = descriptors_to_keep
        self.descriptors = dataset[self.descriptors_to_keep]
        self.labels = dataset[self.result_column_label]

        self.groups = self._group_dataset(self.dataset)

        self.groups = self._generate_pdfs(self.groups)

    def _group_dataset(self, dataset: pd.DataFrame) -> dict:
        """
        Creates a dictionary by grouping reactions with the same chemicals

        Args: 
            dataset: Pandas Dataframe with all data
        Returns:
            defaultdict: Dictionary of groups of chemicals and their descriptors
        """
        groups = {}  # type:dict
        for idx, row in self.dataset.iterrows():
            key = frozenset((row['XXXi0rg1'],
                             row['XXXi0rg1'], row['XXXorg1']))
            # key = frozenset((row['compound_0'],
            #                 row['compound_1'], row['compound_2']))
            if key in groups:
                grp = groups[key]
            else:
                grp = defaultdict(list)

            grp['descriptors'].append(self.descriptors.iloc[idx].values)
            #grp['descriptors'] = np.array(grp['descriptors'])
            grp['result'].append(self.labels.iloc[idx])
            groups[key] = grp

        for grp in groups.values():
            try:
                grp['descriptors'] = np.array(grp['descriptors']).astype(float)
            except:
                print(grp['descriptors'])

        return groups

    def _generate_pdfs(self, groups: dict) -> dict:
        """
        Generates pdfs for a particular reaction
        and joint pdfs for their reaction success or failure
        Args:
            groups: Dictionary of reactions
        Returns:
            groups: Dictionary of reactions with pdfs and joint_pdfs
        """
        for compound, group in groups.items():
            groups[compound]['pdf'] = Pdf(group)
        return groups

    def get_delta_mutual_info(self, candidate: np.array) -> float:
        best_compounds = None
        best_probability = 0.0
        delta_mutual_info = 0.0
        for compounds, group in self.groups.items():
            probability = group['pdf'].prob_feature(candidate)
            if probability > best_probability:
                best_compounds = compounds
                best_probability = probability
        if best_compounds:
            delta_mutual_info = self.groups[best_compounds]['pdf'].delta_mi(
                candidate)
        return delta_mutual_info


if __name__ == "__main__":
    # dataset = pd.read_csv('~/test_data.csv')
    dataset = pd.read_csv('~/test_data_full.csv', low_memory=False)
    dataset_org = dataset.iloc[:6000]
    dataset_new = dataset.iloc[-100:]
    descriptors_to_keep = [
        col for col in dataset.columns if col not in DESCRIPTORS_TO_REMOVE]
    desc_to_keep_testing = [col for col in dataset.columns if col not in DESCRIPTORS_TO_REMOVE] + [
        'boolean_crystallisation_outcome_manual_0']
    dataset_new = dataset_new[desc_to_keep_testing].values
    mi = MutualInformation(
        dataset_org, 'boolean_crystallisation_outcome_manual_0', descriptors_to_keep)

    #print([group['pdf'].descriptor_normal for group in mi.groups.values()])
    print([mi.get_delta_mutual_info(row[:268].astype(float))
           for row in dataset_new])
