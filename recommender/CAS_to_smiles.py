import pandas as pd
import json, itertools, csv, sys, os
import requests
import urllib.request
import time
from csv import writer
from bs4 import BeautifulSoup

'''
Create a dictionary of name and type pairs, so we know what type of reactant it is
'''
def generate_name_type_dictionary(path_to_csv):
    molecule_info = path_to_csv
    molecule_info_df = pd.read_csv(molecule_info)
    # Fill missing values of abbreviation using name
    with open('../sample_data/smiles.csv','a') as fd:
        for x in molecule_info_df['CAS number']:
            url = 'https://cactus.nci.nih.gov/chemical/structure/' + x + '/smiles'
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            print(soup)
            fd.write(str(soup))
            fd.write("\n")

if __name__ == "__main__":
    generate_name_type_dictionary("../sample_data/CG.csv")
    