import pandas as pd
import json, itertools, csv, sys, os

'''
Create a dictionary of name and type pairs, so we know what type of reactant it is
'''
def generate_name_type_dictionary(path_to_csv):
    molecule_info = path_to_csv
    molecule_info_df = pd.read_csv(molecule_info)
    # Fill missing values of abbreviation using name
    # molecule_info_df.loc[molecule_info_df['Abbreviation'].isnull(),'Abbreviation'] = molecule_info_df["Name"]
    # Only use abbreviation and type
    molecule_type_df = molecule_info_df[['smiles',"type (i = inorganic, o=organic, w=water, s=solvent, p=pH)"]]
    molecule_type_df = molecule_type_df.dropna()
    # Create a dictionary of abbreviation and type
    molecule_type_dict = molecule_type_df.set_index('smiles')['type (i = inorganic, o=organic, w=water, s=solvent, p=pH)'].to_dict()

    return molecule_type_dict

def generate_name_smiles_dictionary(path_to_csv):
    molecule_info = path_to_csv
    molecule_info_df = pd.read_csv(molecule_info)
    # Fill missing values of abbreviation using name
    molecule_info_df.loc[molecule_info_df['Abbreviation'].isnull(),'Abbreviation'] = molecule_info_df["Name"]
    # Only use abbreviation and type
    molecule_smiles_df = molecule_info_df[['Abbreviation',"smiles"]]
    molecule_smiles_df = molecule_smiles_df.dropna()
    # Create a dictionary of abbreviation and type
    molecule_smiles_dict = molecule_smiles_df.set_index('Abbreviation')['smiles'].to_dict()

    return molecule_smiles_dict


'''
Generator from old research code, modified to use the dictionary above
'''
def calculate(molecule, ratio, ratio_lookup,molecule_smiles_dict): 
    '''gets the mass from the mole ratio. Ratios sum to 1.0, 
    so we can just multiply.'''
    inv_map = {v: k for k, v in molecule_smiles_dict.items()}
    return ratio_lookup[inv_map[molecule]] * ratio # TODO: is that right? 

def generate_mole_permutations(metal_pair, nonmetals, mole_ratios, ratio_lookup,molecule_smiles_dict):
    combos = []
    for nonmetal in nonmetals:
        if metal_pair[0] == metal_pair[1]:
            for ratios in mole_ratios: #make a different list!
                first_amount = calculate(metal_pair[0], ratios[0] + ratios[1], ratio_lookup, molecule_smiles_dict)
                combos.append([metal_pair[0], 
                        first_amount, 
                        metal_pair[0],
                        first_amount,
                        nonmetal,
                        calculate(nonmetal, ratios[2], ratio_lookup,molecule_smiles_dict)])
        else:
            for ratios in mole_ratios:
                combos.append([metal_pair[0], 
                        calculate(metal_pair[0], ratios[0], ratio_lookup,molecule_smiles_dict), 
                        metal_pair[1], 
                        calculate(metal_pair[1], ratios[1], ratio_lookup,molecule_smiles_dict), 
                        nonmetal,
                        calculate(nonmetal, ratios[2], ratio_lookup,molecule_smiles_dict)])
    return combos

# def generate_row(idx, parameters, parameterstats):
#     result = []
#     for p in range(len(parameterstats)):
#         result.append(parameters[p][idx//parameterstats[p]])
#         idx = idx % parameterstats[p]
#     result += ["no", 2,4,""]
#     return result

def generate_triples_amounts(molecules_file_path,parameters_file_path,output_csv_file_path,molecule_type_dict,molecule_smiles_dict):
    header_row = ["Reaction number","Reactant 1 name","Reactant 1 mass (g)","Reactant 2 name","Reactant 2 mass (g)","Reactant 3 name","Reactant 3 mass (g)","Reactant 4 name","Reactant 4 mass (g)","Reactant 5 name","Reactant 5 mass (g)","Temp (C)","Time (h)","Slow cool","pH (x = unknown)","Leak","purity (0 = no data in notebook, 1 = multiphase; 2 = single phase)","outcome (0 = no data in notebook, 1 = no solid; 2 = noncrystalline/brown; 3 = powder/crystallites; 4 = large single crystals)","Notes"]
    parameters = json.load(open(parameters_file_path)) #list of field names
    molecules = json.load(open(molecules_file_path)) # dictionary of abbrev -> (mass_to_mole)
    metals = list()
    nonmetals = list()

    for mol in molecules:
        try:
            if molecule_type_dict[molecule_smiles_dict[mol]] == "i":
                metals.append(molecule_smiles_dict[mol])
            elif molecule_type_dict[molecule_smiles_dict[mol]] == "o":
                nonmetals.append(molecule_smiles_dict[mol])
        except:
            # Skip all those compunds with unknown type
            continue

    metals = metals[:5]
    nonmetals = nonmetals[:10]
    metal_pairs = itertools.combinations_with_replacement(metals,2)

    old_list = []

    ratio_list = [(4,0,0),(0,4,0),(0,0,4),(3,1,0),(3,0,1),(1,3,0),
            (0,3,1),(1,0,3),(0,1,3),(2,1,1,),(1,2,1),(1,1,2),(2,2,0),(2,0,2)] # this should make the correct triangle
    double_ratio_list = [(4,0,0),(0,0,4),(3,0,1),(1,0,3),(2,0,2)]
    ratio_list = [
            (float(x[0])*0.19+.08, 
                float(x[1])*0.19+0.08, 
                float(x[2])*0.19+0.08) for x in ratio_list] # this converts to mole ratios

    for metal_pair in metal_pairs:
        old_list += generate_mole_permutations(metal_pair, nonmetals, ratio_list, molecules, molecule_smiles_dict)
    
    return old_list


    # with open(output_csv_file_path,"w") as res:
    #     writer = csv.writer(res)
    #     writer.writerow(header_row)
    #     pstats = [len(p) for p in parameters][::-1]
    #     prunning = 1
    #     pstats2 = []
    #     for pc in pstats:
    #         prunning *= pc
    #         pstats2.append(prunning)
    #     pcount = pstats2[-1]
    #     pstats = pstats2[::-1][1:] + [1]
    #     mcount = 0
    #     print("pstats")
    #     print(pstats)
    #     print("pstats2")
    #     print(pstats2)
    #     print("pcount")
    #     print(pcount)
    #     print("parameters")
    #     print(parameters)
    #     for row in old_list:
    #         for idx in range(pcount):
    #             writer.writerow(["%d.%d" % (mcount, idx)] + row + generate_row(idx, parameters, pstats))
    #         mcount += 1
    #     print("---- generate.py: Success!")

def nature_triples_amounts_array_to_json(arr,path_to_json):
    for i in range(len(arr)):
        arr[i] = {"compounds":[arr[i][0],arr[i][2],arr[i][4]],"amounts":[arr[i][1],arr[i][3],arr[i][5]]}

    with open(path_to_json, 'w') as f:
        json.dump(arr, f)


if __name__ == "__main__":
    molecule_type_dict = generate_name_type_dictionary("../sample_data/CG.csv")
    molecule_smiles_dict = generate_name_smiles_dictionary("../sample_data/CG.csv")
    print(molecule_type_dict)
    print(molecule_smiles_dict)
    triples_amounts_array = generate_triples_amounts("../sample_data/mols.json",
                                "../sample_data/parameters.json", "../nature_reactions.csv", 
                                molecule_type_dict, molecule_smiles_dict)
    print(triples_amounts_array)
    nature_triples_amounts_array_to_json(triples_amounts_array,'../sample_data/nature_triples_and_amounts.json')


