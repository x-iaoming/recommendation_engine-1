import random
from chemdescriptor import ChemAxonDescriptorGenerator
from splitter.newsplitter import NewSplitter
import json
from itertools import product, chain
import pandas as pd
import numpy as np
import os
import csv
import subprocess

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

        # print(ph_values)
        cag = ChemAxonDescriptorGenerator(self.compound_set,
                                          descriptor_list,
                                          list(ph_values))
        print(type(output_filename))
        print(output_filename)
        self.descriptor_dataframe = cag.generate(
            str(output_filename), dataframe=True)
        # print(self.descriptor_dataframe)
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
        return self.all_combos_expanded
'''
   A Class that splits the training_data into test and train file, trains a machine learning model using the splitted files,
   and passes the validation_data to 
'''

class MLmodel:
    
    def __init__(self, model_name, training_data, validation_data):
    #validation data is the set of reactions that need to be predicted.     
    #format of validation_data: arff file with output attribute, but all the output values should be set as ?
    #format of training_data: csv file. 
    #validation_data and training_data need to have the exact same attributes
    #current code only supports J48, and SVM. But more models can be added in similar way
    #Sample WEKA configuration from nature paper: https://media.nature.com/original/nature-assets/nature/journal/v533/n7601/extref/nature17439-s4.txt
        if model_name = "J48"
            self.weka_command = "weka.classifiers.trees.J48"
        if model_name = "SVM"
            self.weka_command = "weka.classifiers.functions.SMO"
        else: raise Exception("This model is not recognizable")           

    """
    This function might not work well with subprocess. Might need to manually set the weka path on terminal
    """    
    def set_weka_path(self,class_path = "/home/h205c/Downloads/weka-3-8-3/weka.jar")
        set_weka = "export CLASSPATH={}".format(class_path)
        subprocess.check_output(set_weka, shell=True)
        return
    '''
    This function does test/train split. Original function is in NewSplitter.py
    '''
    def split():
        splitter = NewSplitter()
        splitter.split(self.training_data)
        train = "../train.csv"
        test = "../test.csv"
        #default path to train.csv and test.csv
        #TODO: Convert train.csv and test.csv to arff extension, and do the following data-processing tasks before training
        #Pre-processing tasks: 1. Remove attributes starting with XXX 2. Convert attribute of "outcome" from regression (numeric) to classification {0,1}
        #3. Remove the attribute "purity", which is the second outcome instead of one of the training parameters
        return train,test
 
    '''
    Train a ml model using the test and train files generated from split()
    '''

    def train(self,path_to_model_file):
        self.set_weka_path()
        train_arff, test_arff = self.split()

        command = "java {} -d {} -t {} -T {} -p 0"
        .format(self.weka_command, path_to_model_file, train_arff, test_arff)
        
        puk_omega = 1
        puk_sigma = 1
        if self.model_name = "SVM"
           kernel = "weka.classifiers.functions.supportVector.Puk"
           command = "java {} -d {} -t {} -T {} -K {} -O {} -S {} -p 0"
        .format(self.weka_command, path_to_model_file, train_arff, test_arff, kernel,puk_omega, puk_sigma)

        subprocess.check_output(command, shell=True)

        return path_to_model_file

    '''
    Run model that is already trained and make predictions
    @Args:
        model_file: trained model with .model extension
        result_path: validation_data + predicted outcomes. It is hardcoded now. Should be adjusted for different servers
    @return: a list of 0 and 1
    '''
    def run_trained_model(self,result_path = "/home/h205c/recommendation_engine/prediction4.csv"):
        
        self.set_weka_path()
        model_file = self.train()
        command = "java {} -T {} -l {} -p 0 1> {}".format(
          self.weka_command, self.validation_data, model_file, result_path) 
        subprocess.check_output(command, shell=True)
        # Read weka prediction output
        prediction_index = 2
        ordConversion = lambda s: int(s.split(':')[1])
        with open(result_path, "r") as f:
        # Discard the headers and ending line.
            raw_lines = f.readlines()[5:-1]
            raw_predictions = [line.split()[prediction_index]
                           for line in raw_lines]
            predictions = [ordConversion(
                prediction) for prediction in raw_predictions]
            return predictions


    
    #TODO: Convert reaction_dataframe to proper arff file 
    def sieve(self, reaction_dataframe, descriptor_whitelist=[]):
        """
        Passes sampled reactions through an ML model. Only reactions predicted as 
        sucessful are returned
        """

        if not descriptor_whitelist:
            self.runmodel(reaction_dataframe)
        else:
            self.runmodel(reaction_dataframe[descriptor_whitelist])
        reaction_dataframe['prediction'] = self.result


if __name__ == "__main__":

    # Running order: generate(), generateDescriptor(), expandedgrid()
    
    os.environ['CXCALC_PATH'] = '/home/h205c/chemaxon/bin'
    #os.environ['CXCALC_PATH'] = '/Applications/MarvinSuite/bin'

    turl = "../sample_data/triples_and_amounts.json"
    gurl = "../sample_data/grid_params.json"
    test = Generator(turl, gurl)
    #test.generate_grid()

    desf = "../sample_data/descriptors_list.json"
    outputfile = "/home/h205c/recommendation_engine/sample_data/descriptoroutput.txt"
   # test.generate_descriptors(desf,outputfile)

   # combos = test.generate_expanded_grid()
  #  print(combos)
    """
    Running Machine Learning Model J48
    """
    mlmodel = MLmodel("J48", training_data, validation_file)
    training_data = '/home/h205c/recommendation_engine/sample_data/nature17439-s2.csv'
    validation_file = "/home/h205c/recommendation_engine/validation.arff"
    model_file = "/home/h205c/recommendation_engine/j48.model"

    result = mlmodel.runmodel()
    print(result)
    #df = mlmodel.sieve(test.all_combos_expanded)
    #print(df)
    # print(df)
    