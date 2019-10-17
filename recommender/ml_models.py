from NewSplitter import NewSplitter
import os
import subprocess
import pandas as pd
import shutil


class WekaModel:
    """
    Machine learning model class that preprocesses data, trains a model, 
    makes predictions on validation data, and return asuccessful reactions
    as a dataframe
    """

    def __init__(self,
                 model_name,
                 all_data_path,
                 validation_data_path,
                 weka_path="/home/h205c/Downloads/weka-3-8-3/weka.jar",
                 descriptor_whitelist=[]):
        """
        It initializes with all_data_path and validation_data_path, which are csv files. They need to have the exact same attributes.
        """
        self.all_data = all_data_path
        self.all_data_df = pd.read_csv(all_data_path, low_memory=False)
        self.model_name = model_name
        self.weka_path = weka_path
        self.validation_data = validation_data_path
        self.validation_data_df = pd.read_csv(
            validation_data_path, low_memory=False)

        self.descriptor_whitelist = descriptor_whitelist

        self.java_command = 'java'

        if self.model_name == "J48":
            self.weka_command = "weka.classifiers.trees.J48"
        elif self.model_name == "SVM":
            self.weka_command = "weka.classifiers.functions.SMO"
        elif self.model_name == "LogisticRegression":
            self.weka_command = "weka.classifiers.functions.Logistic"
        elif self.model_name == "KNN":
            self.weka_command = "weka.classifiers.lazy.IBk"
        elif self.model_name == "RandomForest":
            self.weka_command = "weka.classifiers.meta.FilteredClassifier"
        else:
            raise Exception("This model is not recognizable")

    def _convert(self, file_path):
        """
        This function takes in a csv file, converts it to arrf file, and returns the path to the arff file
        Used in 'init' and 'train'
        """
        if file_path[-4:] != '.csv':
            raise Exception('Please input a CSV file')

        abs_path = os.path.abspath(file_path)
        print(abs_path[:-4]+'.arff')

        command = '{} -cp {} weka.core.converters.CSVLoader {} > {}.arff'.format(
            self.java_command, self.weka_path, abs_path, abs_path[:-4])
        subprocess.call(command, shell=True)

        # Convert outcome to binary
        command = '{0} -cp {1} weka.filters.unsupervised.attribute.NumericToBinary -R last -i {2}.arff -o {2}_new.arff'.format(
            self.java_command, self.weka_path, file_path[:-4])
        subprocess.call(command, shell=True)

        shutil.copy2('{}_new.arff'.format(
            file_path[:-4]), '{}.arff'.format(file_path[:-4]))
        os.remove('{}_new.arff'.format(file_path[:-4]))
        return file_path[:-4]+'.arff'

    def train(self, path_to_model_file=None, k=1, i=100):
        """
        This function splits the data set and writes train and test files
        It then trains and writes a model
        k is the number of neigbhors for knn algorithm
        i is the number for random forest algorithm
        """
        # TODO: Filters out descriptors not in the whitelist
        # self.filter(self.all_data,descriptor_whitelist)
        # self.filter(self.validation_data,descriptor_whitelist)

        # Split all data to train and test files, NewSplitter() imported from separate file
        splitter = NewSplitter()
        splitter.split(
            self.all_data, self.all_data_df[self.descriptor_whitelist])

        # Set and run weka model commands
        if not path_to_model_file:
            self.path_to_model_file = self.model_name+'.model'
        train_arff = self._convert("../sample_data/train.csv")
        test_arff = self._convert("../sample_data/test.csv")

        if self.weka_command == "weka.classifiers.functions.SMO":
            # SVM
            command = "{} -cp {} {} -d {} -t {} -T {} -K 'weka.classifiers.functions.supportVector.Puk -O 0.5 -S 7' -p 0".format(
                                                                   self.java_command,
                                                                   self.weka_path,
                                                                   self.weka_command,
                                                                   self.path_to_model_file,
                                                                   train_arff, test_arff)
        elif self.weka_command == "weka.classifiers.meta.FilteredClassifier":
            # RandomForest
            command = "{} -cp {} {} -d {} -t {} -T {} -p 0 -W weka.classifiers.trees.RandomForest -- -I {}".format(
                                                                   self.java_command,
                                                                   self.weka_path,
                                                                   self.weka_command,
                                                                   self.path_to_model_file,
                                                                   train_arff, test_arff,i)
        else:
            # J48, LogisticRegression
            command = "{} -cp {} {} -d {} -t {} -T {} -p 0".format(self.java_command,
                                                                   self.weka_path,
                                                                   self.weka_command,
                                                                   self.path_to_model_file,
                                                                   train_arff, test_arff)
            if self.weka_command == "weka.classifiers.lazy.IBk":
                command.append(" -K {}".format(k))

        subprocess.check_output(command, shell=True)

        # if self.model_name == "SVM":
        #     puk_omega = 1
        #     puk_sigma = 1
        #     kernel = "weka.classifiers.functions.supportVector.Puk"
        #     command = "java {} -d {} -t {} -T {} -K {} -O {} -S {} -p 0"
        #      .format(self.weka_command, path_to_model_file, train_arff, test_arff, kernel,puk_omega, puk_sigma)

    def _read_weka_output(self, result_path):
        """
        This function reads a weka prediction output file and stores the prediction results as a list of 0 and 1
        Used in 'predict'
        """
        prediction_index = 2
        def ordConversion(s): return int(s.split(':')[1])
        with open(result_path, "r") as f:
            raw_lines = f.readlines()[5:-1]
            raw_predictions = [line.split()[prediction_index]
                               for line in raw_lines]
            self.predictions = [ordConversion(
                prediction) for prediction in raw_predictions]

    def read_actual(self, result_path):
        """
        This function reads a weka prediction output file and stores the prediction results as a list of 0 and 1
        Used in 'predict'
        """
        prediction_index = 1
        def ordConversion(s): return int(s.split(':')[1])
        with open(result_path, "r") as f:
            raw_lines = f.readlines()[5:-1]
            raw_predictions = [line.split()[prediction_index]
                               for line in raw_lines]
            self.actual = [ordConversion(
                prediction) for prediction in raw_predictions]
        

    def predict(self, result_path="../sample_data/prediction.csv"):
        """
        This function runs the model that is already trained and stores the prediction results
        """
        # Make sure the 'train' function was called
        if not self.path_to_model_file:
            raise Exception("Please train a model first")

        validation_filtered = self.validation_data_df[self.descriptor_whitelist]
        validation_filtered.to_csv(
            self.validation_data[:-4]+'_filtered.csv', index=False)
        # Run prediction
        command = "{} -cp {} {} -T {} -l {} -p 0 1> {}".format(self.java_command,
                                                               self.weka_path,
                                                               self.weka_command,
                                                               self._convert(
                                                                   self.validation_data[:-4]+'_filtered.csv'),
                                                               self.path_to_model_file, result_path)
        subprocess.check_output(command, shell=True)

        # Convert weka prediction to a list of results
        self._read_weka_output(result_path)
        self.read_actual(result_path)

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

    
