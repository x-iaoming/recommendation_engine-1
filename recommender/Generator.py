import random
from chemdescriptor import ChemAxonDescriptorGenerator
from NewSplitter import NewSplitter
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
    #TODO: Fixed bug. When I ran this function, it shows that "can't convert list to str implicitly" fpr line 110.

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

        self.descriptor_dataframe = cag.generate(
            output_filename, dataframe=True)
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


class MLmodel:
    """
    Machine learning model class that preprocesses data, trains a model, 
    makes predictions on validation data, and return asuccessful reactions
    as a dataframe
    """
    def __init__(self, 
                 model_name, 
                 all_data_path, 
                 validation_data_path, 
                 weka_path="/home/h205c/Downloads/weka-3-8-3/weka.jar"):
        """
        It initializes with all_data_path and validation_data_path, which are csv files. They need to have the exact same attributes.
        """
        self.all_data = all_data_path    
        self.model_name = model_name
        self.weka_path = weka_path
        self.validation_data = validation_data_path

        if self.model_name == "J48":
            self.weka_command = "weka.classifiers.trees.J48"
        elif self.model_name == "SVM":
            self.weka_command = "weka.classifiers.functions.SMO"
        else: 
            raise Exception("This model is not recognizable")           
 
    def _convert(self,file_path):
        """
        This function takes in a csv file, converts it to arrf file, and returns the path to the arff file
        Used in 'init' and 'train'
        """
        if file_path[-4:] != '.csv':
            raise Exception('Please input a CSV file')

        command = 'java -cp '+self.weka_path+' weka.core.converters.CSVLoader '+file_path+' > '+file_path[:-4]+'.arff'
        subprocess.call(command, shell=True)
        return file_path[:-4]+'.arff'

    def train(self,path_to_model_file=None,descriptor_whitelist=[]):
        """
        This function splits the data set and writes train and test files
        It then trains and writes a model
        """
        # TODO: Filters out descriptors not in the whitelist
        # self.filter(self.all_data,descriptor_whitelist)
        # self.filter(self.validation_data,descriptor_whitelist)

        # Split all data to train and test files, NewSplitter() imported from separate file
        splitter = NewSplitter()
        splitter.split(self.all_data)

        # Set and run weka model commands
        if not path_to_model_file:
            self.path_to_model_file = self.model_name+'.model'
        train_arff = self._convert("../train.csv")
        test_arff = self._convert("../test.csv")
        command = "java -cp {} {} -d {} -t {} -T {} -p 0".format(
            self.weka_path, self.weka_command, self.path_to_model_file, train_arff, test_arff)
        
        subprocess.check_output(command, shell=True)
        
        # if self.model_name == "SVM":
        #     puk_omega = 1
        #     puk_sigma = 1
        #     kernel = "weka.classifiers.functions.supportVector.Puk"
        #     command = "java {} -d {} -t {} -T {} -K {} -O {} -S {} -p 0"
        #      .format(self.weka_command, path_to_model_file, train_arff, test_arff, kernel,puk_omega, puk_sigma)
    
    def _read_weka_output(self,result_path):
        """
        This function reads a weka prediction output file and stores the prediction results as a list of 0 and 1
        Used in 'predict'
        """
        prediction_index = 2
        ordConversion = lambda s: int(s.split(':')[1])
        with open(result_path, "r") as f:
            raw_lines = f.readlines()[5:-1]
            raw_predictions = [line.split()[prediction_index]
                           for line in raw_lines]
            self.predictions = [ordConversion(
                prediction) for prediction in raw_predictions]

    def predict(self,result_path="/home/h205c/recommendation_engine/prediction.csv"):
        """
        This function runs the model that is already trained and stores the prediction results
        """ 
        # Make sure the 'train' function was called
        if not self.path_to_model_file:
            raise Exception("Please train a model first")

        # Run prediction
        command = "java -cp {} {} -T {} -l {} -p 0 1> {}".format(
          self.weka_path, self.weka_command, self._convert(self.validation_data), self.path_to_model_file, result_path) 
        subprocess.check_output(command, shell=True)
        
        # Convert weka prediction to a list of results 
        self._read_weka_output(result_path)
    
    def sieve(self):
        """
        This function returns a dataframe of successful reactions
        """
        # Make sure the 'predict' function was called
        if not self.predictions:
            raise Exception("No predictions found")

        validation_dataframe = pd.read_csv(self.validation_data)
        validation_dataframe['prediction'] = self.predictions
        return validation_dataframe.loc[validation_dataframe['prediction'] == 1]



if __name__ == "__main__":

    # Running order: generate(), generateDescriptor(), expandedgrid()
   #  os.environ['CXCALC_PATH'] = '/home/h205c/chemaxon/bin'
   #  # os.environ['CXCALC_PATH'] = '/Applications/MarvinSuite/bin'
   #  turl = "../sample_data/triples_and_amounts.json"
   #  gurl = "../sample_data/grid_params.json"
   #  test = Generator(turl, gurl)
   #  #test.generate_grid()
   #  desf = "../sample_data/descriptors_list.json"
   #  outputfile = "/home/h205c/recommendation_engine/sample_data/descriptoroutput.txt"
   # # test.generate_descriptors(desf,outputfile)
   #  combos = test.generate_expanded_grid()
   #  print(combos)




    """
    Testing MLmodel class only. After calling train,predict,and sieve: 
    generate train.csv>test.csv>train.arff>test.arff>J48.model>predict.csv
    return a data frame of successful reactions

    TODO: 
    Automatic data preprocessing
    1. Remove attributes starting with XXX and implement whitelist in train()
    2. Convert attribute of "outcome" from regression (numeric) to classification {0,1} 
    3. Remove the attribute "purity", which is the second outcome instead of one of the training parameters 
    4. For validation data, change 'outcome' to '?' for weka to make predictions (see weka documentation)
    """
    all_data = '/home/h205c/recommendation_engine/sample_data/nature17439-s2.csv'
    validation_file = "/home/h205c/recommendation_engine/validation.csv"
    mlmodel = MLmodel("J48", all_data, validation_file)
    mlmodel.train()
    mlmodel.predict()
    print(mlmodel.sieve())




    #df = mlmodel.sieve(test.all_combos_expanded)
    #print(df)
    # print(df)
    