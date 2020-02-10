import random
from chemdescriptor import ChemAxonDescriptorGenerator

import json
from itertools import product, chain
import pandas as pd
import numpy as np
import os
import csv
from ml_models import WekaModel

os.environ['CXCALC_PATH'] = '/home/h205c/chemaxon/bin'




class Generator:
    """
    This generator class helps generate a grid of possible reactions
    """

    def __init__(self, compound, params):
        """
        Initializes the generator
        Args:
            compound:    Path to json file with dictionaries of compounds and amounts/or dict
            params:      Path to json file with a dictionary of parameters/or dict
        """

        if isinstance(compound, str):
            compounds = open(compound)
            self.compounds_data = json.load(compounds)
        elif isinstance(compound, (list, set, tuple)):
            self.compounds_data = compound
        else:
            raise Exception(
                "'compound' should be a path or list. Found: {}".format(type(compound)))

        if isinstance(params, dict):
            self.params_grid_data = params
        elif isinstance(params, str):
            self.params_grid_data = json.load(open(params, 'r'))
        else:
            raise Exception(
                "'params' should be a path or dict. Found: {}".format(type(params)))
        self.all_combos = []
        self.total_compounds = len(self.compounds_data[0]['compounds'])
        self.compound_set = []

        for experiment in self.compounds_data:
            for compound in experiment['compounds']:
                self.compound_set.append(compound)

    def generate_grid(self):
        """
        Generates all combinitions of compounds, their amounts, and parameters
        and store them in the 2-d array called self.all_combos
        """
        # Get a list of parameters values
        list_params = list(self.params_grid_data.values())

        # Generate a list of all combos for params
        params_combos = list(product(*list_params))
        names = self._generate_column_names()

        # Loop over the array of dictionary for the compounds and amounts:
        for experiment in self.compounds_data:
            for params in params_combos:
                compounds = experiment['compounds']
                amounts = experiment['amounts']
                self.all_combos += [chain(compounds, amounts,
                                          params)]
        self.all_combos = pd.DataFrame.from_records(self.all_combos)
        self.all_combos.columns = names

        return self.all_combos

    def _generate_column_names(self):
        """
        Generates column names for the final dataframe based on the input values
        Used in self.generate()
        """
        names = []
        # Assuming same number of compounds for every reaction!

        names = ['compound_{}'.format(i) for i in range(self.total_compounds)]
        names += ['compound_{}_amount'.format(i)
                  for i in range(self.total_compounds)]
        for grid_param in self.params_grid_data.keys():
            names.append(grid_param)

        return names

    # TODO: Fixed bug. When I ran this function, it shows that "can't convert list to str implicitly" fpr line 110.

    def generate_descriptors(self, descriptor_list, output_filename):
        """
        Generates descirptors from a file of smile codes and the desired descriptors,
        and stores the output as a csv file
        Args:
            input_molecule_file_path:   Path to ip molecules
            descriptor_file_path:       Path to ip descriptors
            ph_values:                  List of pH values at which to calculate descriptors
            command_stems:              Dictonary of descriptors and its command stem
            ph_command_stems:           Dict of pH related descriptors and command stems
        """

        #smiles = '../sample_data/test_foursmiles.smi'

        ph_values = self.params_grid_data['reaction_pH']

        print(self.compound_set)
        cag = ChemAxonDescriptorGenerator(self.compound_set,
                                          descriptor_list,
                                          list(ph_values))

        self.descriptor_dataframe = cag.generate(output_filename, dataframe=True)
        print(self.descriptor_dataframe)
        # print(
        # self.descriptor_dataframe[['Compound'] + [col for col in self.descriptor_dataframe.columns if ('7' in col)]])

    def _generate_expanded_column_names(self):
        """
        Generates column names for dataframe in expaneded grid
        Used in self.generate_expanded_grid()
        """

        names = []
        # Get names of the descriptors
        des_names = [column for column in self.descriptor_dataframe][1:]

        # Generate expanded descriptor names for each compound
        for i in range(self.total_compounds):
            for des_name in des_names:
                name = 'compund_{}_{}'.format(i, des_name)
                names.append(name)

        return names

    def _generate_expanded_grid_helper(self, iter_no):
        """
        This helper function generates expaneded descriptors for one reaction
        """
        temp = []
        for i in range(self.total_compounds):
            # get the smile code of the ith compound in the reaction
            smile = self.all_combos.iloc[iter_no][i]
            # find the respective expanded descriptors
            compound_i_desc = self.dic[smile]
            # combining the expanded descriptors for every ith compound in the reaction
            temp += compound_i_desc

        return temp

    def generate_expanded_grid(self):
        """
        This function generates expaneded descriptors for all reactions
        """

        # Create a dictionary to get expanded descriptors for compounds
        self.dic = {}
        df = self.descriptor_dataframe
        desc_len = df.shape[0]
        for i in range(desc_len):
            value = df.iloc[i].tolist()
            self.dic[df.iloc[i][0]] = value[1:]

        expanded_grid = []
        # iterate over n reactions in all_combos
        for n in range(self.all_combos.shape[0]):
            # get and combine expanded descriptors for every nth reaction
            expanded_grid += [self._generate_expanded_grid_helper(n)]

        # convert the expanded grid to a dataframe with appropriate names
        expanded_grid_df = pd.DataFrame(
            expanded_grid, columns=self._generate_expanded_column_names())

        # merge the original combos with the expaneded grid
        self.all_combos_expanded = pd.concat(
            [self.all_combos, expanded_grid_df], axis=1)

        self.all_combos_expanded.to_csv("../sample_data/combos.csv")
        return self.all_combos_expanded

if __name__ == "__main__":

    os.environ['CXCALC_PATH'] = '/home/h205c/chemaxon/bin'
    #os.environ['CXCALC_PATH'] = '/Applications/MarvinSuite/bin'

    # Generate reactions from nature paper
    gurl = "../sample_data/nature_grid_params.json"
    turl = "../sample_data/nature_triples_and_amounts.json"
    
    # Generate possible reactions from given parameters
    # turl = "../sample_data/triples_and_amounts.json"
    # gurl = "../sample_data/grid_params.json"
    test = Generator(turl, gurl)
    test.generate_grid()
    desf = "../sample_data/descriptors_list.json"
    outputfile = "../sample_data/descriptoroutput.txt"
    test.generate_descriptors(desf, outputfile)
    
    # Print generator output(saved to file as well)
    combos = test.generate_expanded_grid()

    # Below is working for direct import of csv files 
    # all_data = "../nature_reactions.csv"
    # validation_file = "../sample_data/validation.csv"
    all_data = '../sample_data/combos.csv'
    validation_file = "../sample_data/combos.csv"

    # Filter whitelist
    
    all_data_df = pd.read_csv(all_data, low_memory=False)
    whitelist = [col for col in all_data_df.columns if 'XXX' not in col]


    # Run ML model
    # Avaliable models are SVM, J48, LogisticRegression, KNN, RandomForest
    mlmodel = WekaModel("SVM", all_data, validation_file,descriptor_whitelist=whitelist)
    # optional parameter for train():
    # k for number of neighbors for KNN, i for randomforest number
    mlmodel.train()
    mlmodel.predict()
    success = mlmodel.sieve()
    validation_prediction = mlmodel.validation_prediction

    # Output predictions
    correct = 0
    for i in range(len(mlmodel.predictions)):
    	if mlmodel.predictions[i] == mlmodel.actual[i]:
    		correct +=1
    print("Accuracy:"+"{:.3%}".format(correct/len(mlmodel.predictions)))

    from mutual_information import *
    # DESCRIPTORS_TO_REMOVE = ['compound_0_amount_grams', 'compound_1_amount_grams',
    #                      'compound_2_amount_grams', 'labGroup', 'notes', 'compound_0_role',
    #                      'compound_1_role', 'compound_2_role', 'compound_0', 'compound_1',
    #                   'compound_2', 'compound_0_amount', 'compound_1_amount',
    #                      'compound_2_amount']

    compound_column_labels = ['XXXi0rg1', 'XXXi0rg2', 'XXXi0rg3']

    # convert bool to int and ? to 0
    all_data_df['slowCool']= pd.to_numeric(all_data_df['slowCool'], errors='coerce').fillna(0)
    all_data_df['leak']= pd.to_numeric(all_data_df['leak'], errors='coerce').fillna(0)

    # calculate mutual information on all_data_df
    mi = MutualInformation(
       all_data_df, 'outcome', compound_column_labels, whitelist)

    # convert bool to int 
    success['slowCool'] = pd.to_numeric(success['slowCool'], errors='coerce').fillna(0)
    success['leak'] = pd.to_numeric(success['leak'], errors='coerce').fillna(0)

    # calculate the change in MI and get top reactions
    recommended_reactions = mi.get_recommended_reactions(20,success[whitelist].values)
    print(recommended_reactions)



