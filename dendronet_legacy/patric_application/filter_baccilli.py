import pandas as pd

reduced_features = pd.read_csv('data_files/0_bac_PATRIC_sp_gene.csv')

labels = pd.read_csv('data_files/bacilli_PATRIC_genome_amr.csv', dtype=str)

# genome_id = row[patric_id_index][4:].split('.peg')[0]  # extracting the genome ID from patric ID - ugly af
features_ids = list(set((reduced_features['PATRIC ID'])))
features_ids = [feat_id[4:].split('.peg')[0] for feat_id in features_ids]



print('done')