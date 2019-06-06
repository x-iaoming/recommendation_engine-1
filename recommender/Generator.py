import random
from chemdescriptor import ChemAxonDescriptorGenerator
import json
from itertools import product, chain
import pandas as pd
import numpy as np
import os
import csv


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

    def generate(self):
        """
        Generates all combinitions of compounds, their amounts, and parameters
        and store them in the 2-d array called self.all_combos
        """
        # Get a list of parameters values
        list_params = list(self.params_grid_data.values())

        # Generate a list of all combos for params
        params_combos = list(product(*list_params))
        names = self.generate_column_names()

        # Loop over the array of dictionary for the compounds and amounts:
        for experiment in self.compounds_data:
            for params in params_combos:
                compounds = experiment['compounds']
                #amounts = list(experiment.values())[1]
                amounts = experiment['amounts']
                self.all_combos += [chain(compounds, amounts,
                                          params)]
        self.all_combos = pd.DataFrame.from_records(self.all_combos)
        self.all_combos.columns = names
        #print(names)
        #print(self.all_combos.head())

    def generate_column_names(self):
        """
        Generates column names for the final dataframe based on the input values
        Used in self.generate()
        """
        names = []
        # Assuming same number of compounds for every reaction!
        total_compounds = len(self.compounds_data[0]['compounds'])
        names = ['compound_{}'.format(i) for i in range(total_compounds)]
        names += ['compound_{}_amount'.format(i)
                  for i in range(total_compounds)]
        for grid_param in self.params_grid_data.keys():
            names.append(grid_param)

        return names

    def generate_descriptors(self, descriptor_list):
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

        smiles = '../sample_data/test_foursmiles.smi'

        ph_values = self.params_grid_data['reaction_pH']

        #print(ph_values)
        cag = ChemAxonDescriptorGenerator(smiles,
                                          descriptor_list,
                                          list(ph_values))
        self.descriptor_dataframe = cag.generate('opnew.csv', dataframe=True)
        #print(self.descriptor_dataframe)
        #print(
            #self.descriptor_dataframe[['Compound'] + [col for col in self.descriptor_dataframe.columns if ('7' in col)]])
    
    def generate_expanded_column_names(self):
        """
        Generates column names for dataframe in expaneded grid
        Used in self.generate_expanded_grid()
        """

        names = []
        # Get names of the descriptors
        des_names = [column for column in self.descriptor_dataframe][1:]
        # Assuming same number of compounds for every reaction!
        total_compounds = len(self.compounds_data[0]['compounds'])

        # Generate expanded descriptor names for each compound
        for i in range(total_compounds):
            for des_name in des_names:
                name = 'compund_{}_{}'.format(i,des_name)
                names.append(name)

        return names

    def generate_expanded_grid_helper(self,iter_no):
        """
        This helper function generates expaneded descriptors for one reaction
        """

        # df to search for the expanded descriptors based on compound smile code 
        df = self.descriptor_dataframe
        # number of compounds to find respective descriptors
        total_compounds = len(self.compounds_data[0]['compounds'])

        temp = []
        for i in range(total_compounds):
            # get the smile code of the ith compound in the reaction
            smile = self.all_combos.iloc[iter_no][i]
            # find the matching descriptors for the ith compound in df
            compound_i_desc_df = df[df['Compound'] == smile][:]
            # convert them to a list
            compound_i_desc = compound_i_desc_df.values.tolist()[0][1:]
            # combining the expanded descriptors for every ith compound in the reaction
            temp += compound_i_desc

        return temp

    def generate_expanded_grid(self):
        """
        This function generates expaneded descriptors for all reactions
        """

        expanded_grid = []
        # iterate over n reactions in all_combos
        for n in range(self.all_combos.shape[0]):
            # get and combine expanded descriptors for every nth reaction
            expanded_grid += [self.generate_expanded_grid_helper(n)]

        # convert the expanded grid to a dataframe with appropriate names
        expanded_grid_df = pd.DataFrame(expanded_grid,columns = self.generate_expanded_column_names())

        # merge the original combos with the expaneded grid
        self.all_combos_expanded = pd.concat([self.all_combos,expanded_grid_df],axis = 1)

            


        # desc_dict = self.desc_dict()

        # self.descriptor_dataframe
        # total_compounds = len(self.compounds_data[0]['compounds'])
        # for i in range(total_compounds):
        #     self.all_combos.columns[i]

        # self.dic = {}
        # for row in reader:
        #     self.dic[row[0]] = row[1:]
        # # loops through all_combos and adds the descriptors in each reaction
        # for reactionindex in range(len(self.all_combos)):
        #     for i in range(3):
        #         # Checks to see the compound names are in the first or second column.
        #         if isinstance(self.all_combos[reactionindex][1][i], str):
        #             compound = self.all_combos[reactionindex][1][i]
        #         else:
        #             compound = self.all_combos[reactionindex][0][i]
        #         # Extracts the descriptors of each compound
        #         descip = self.dic[compound]
        #         self.all_combos[reactionindex] += [descip]

        # name_params = list(self.params_grid_data.keys())
        # # set the names for the 13 columns in the expanded grid
        # headers = ["compounds", "amounts"] + name_params + \
        #     ["C1descriptor", "C2descriptor", "C3descriptor"]

        # newdf = pd.DataFrame(self.all_combos, columns=headers)

        # return newdf

    def sieve(self, desired_des, predictions):
        index_of_undesireddic = set()
        desired_des = set(desired_des)

        for desc in range(len(self.dic['Compound'])):
            if self.dic['Compound'][desc] not in desired_des:
                index_of_undesireddic.add(desc)

        for i in range(len(self.all_combos)):
            row = self.all_combos[i]
            for compound in row[-3:]:
                if len(compound) == len(desired_des):
                    continue
                for j in range(len(compound), -1, -1):
                    if j in index_of_undesireddic:
                        del compound[j]

        headers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

        newdf = pd.DataFrame(self.all_combos, columns=headers)
        mlmodel.runmodel(self.all_combos)
        result = mlmodel.result  # a list of 0 and 1
        newdf['result'] = result

        return newdf.loc[newdf['result'] == 1]


class MLmodel:

    def runmodel(self, reactions):
        result = []
        for reaction in range(len(reactions)):
            if random.random() < 0.7:
                num = 0
            else:
                num = 1
            result.append(num)
        self.result = result


if __name__ == "__main__":

    # Running order: generate(), generateDescriptor(), expandedgrid()
    os.environ['CXCALC_PATH'] = '/home/h205c/chemaxon/bin'

    turl = "../sample_data/triples_and_amounts.json"
    gurl = "../sample_data/grid_params.json"
    test = Generator(turl, gurl)
    test.generate()
    print(test.all_combos)

    desf = '../sample_data/descriptors_list.json'
    test.generate_descriptors(desf)
    print(test.descriptor_dataframe)

    test.generate_expanded_grid()
    print(test.all_combos_expanded)

    # csvfile = "opnew.csv"
    # desc = test.expand_grid(csvfile)
    # mlmodel = MLmodel()
    # df = test.sieve(['vanderwaals_nominal', 'vanderwaals_ph7',
    #                 'asa_nominal', 'asa_ph7'], mlmodel)
    # print(df)
