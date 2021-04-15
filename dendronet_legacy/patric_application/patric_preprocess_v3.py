import os
import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing the patric data')
    parser.add_argument('--antibiotics', type=str, default=['chloramphenicol'], metavar='A',
                        help='Antibiotics of interest')
    parser.add_argument('--properties', type=str, default=['Antibiotic Resistance', 'Virulence Factor'], metavar='P',
                        help='properties of retrieved gene features (default: [Antibiotic Resistance, Virulence Factor])')
    parser.add_argument('--feature-folder', type=str, default='patric_cli/sp_genes/chloramphenicol', metavar='F',
                        help='name of folder containing feature data')
    parser.add_argument('--label-folder', type=str, default='patric_cli', metavar='L',  # todo: replace these with the new files from patric_cli.py
                        help='name of folder containing label data(default: data_files/genome_ids)')
    parser.add_argument('--only-annotated', type=bool, default=True, metavar='OA',
                        help='Only samples with all drugs annotated')
    parser.add_argument('--output-file', type=str, default='chloramphenicol_all_samples.csv', metavar='O',
                        help='name of the output file')
    args = parser.parse_args()

    # collecting all the genome IDs of relevant training samples

    genome_ids = set()

    for file in os.listdir(args.label_folder):
        if file.split('.')[0] in args.antibiotics:
            df = pd.read_csv(os.path.join(args.label_folder, file), sep='\t', dtype=str)
            df = df[df['genome_drug.resistant_phenotype'].notnull()]
            genome_ids = genome_ids.union(set(df['genome_drug.genome_id']))

    spgene_files = list()
    feature_df = None
    # collecting the relevant special gene sets
    for filename in os.listdir(args.feature_folder):
        if filename.split('_spgene')[0] in genome_ids:  # extracts the genome id and confirms it is a training sample
            spgene_files.append(filename)
            df = pd.read_csv(os.path.join(args.feature_folder, filename), dtype=str, sep='\t')
            df = df.loc[df['property'].isin(args.properties)]
            if feature_df is None:
                feature_df = df
            else:
                feature_df = pd.concat([feature_df, df], ignore_index=True)

    # getting the set of classifications that will serve as a feature vector
    classifications = list(set([feat for feat in list(feature_df['classification']) if pd.notna(feat)]))
    num_classifications = len(classifications)
    print('Number of gene family classifications: ' + str(num_classifications))

    # building our list of genomes and associated resistance label vector, feature vector
    col_names = ['ID', 'Antibiotics', 'Phenotype', 'Annotations', 'Features']
    samples = list()
    for gen_id in genome_ids:
        sample = [gen_id, args.antibiotics, [0 for _ in range(len(args.antibiotics))],
                  [False for _ in range(len(args.antibiotics))], [0.0 for _ in range(num_classifications)]]
        # filling the label vector, noting if actually annotated in the data or default-filled
        for ab_index in range(len(args.antibiotics)):
            curr_ab = args.antibiotics[ab_index]
            ab_phenotypes = pd.read_csv(os.path.join(args.label_folder, str(curr_ab + '.csv')), dtype=str, sep='\t')
            ab_phenotypes = ab_phenotypes[ab_phenotypes['genome_drug.resistant_phenotype'].notnull()]
            for index, row in ab_phenotypes.iterrows():
                if row['genome_drug.genome_id'] == gen_id:
                    sample[3][ab_index] = True  # magic number for annotation vector
                    if row['genome_drug.resistant_phenotype'] == 'Resistant' or row['genome_drug.resistant_phenotype'] == 'Non-susceptible': #todo: decide how to handle this, only relevant for beta-lactam
                        sample[2][ab_index] = 1  # magic number for phenotype vector
                    break
        # filling the feature count vector

        sample_features = [feat for feat in list(feature_df.loc[feature_df['genome_id'] == gen_id]['classification']) if pd.notna(feat)]
        for feat in sample_features:
            sample[-1][classifications.index(feat)] += 1.0
        if not (args.only_annotated and False in sample[3]):
            samples.append(sample)
            print('Sample: ' + str(sample))

    output_file = os.path.join('data_files', args.output_file)
    pd.DataFrame.from_records(samples, columns=col_names).to_csv(output_file, index=False)
    print(str('Preprocessing of ' + str(len(samples)) + ' samples done and saved to ' + output_file))
