import os
import wget
import time
import pandas as pd
# example final url:ftp://ftp.patricbrc.org/genomes/511145.12/511145.12.PATRIC.spgene.tab

# antibiotics = ['moxifloxacin', 'azithromycin', 'clarithromycin', 'clindamycin', 'ceftriaxone']
# antibiotics = ['betalactam']

# antibiotics = ['betalactam', 'ciprofloxacin', 'cloramphenicol', 'cotrimoxazole',
#                'fusidicacid', 'gentamicin', 'rifampin', 'trimethoprim', 'vancomycin']

# antibiotics = ['trimethoprim', 'tetracycline', 'isoniazid', 'ethambutol', 'streptomycin']
antibiotics = ['chloramphenicol']

# ciprofloxacin
# rifampin
# erythromycin
# gentamicin
# chloramphenicol
# kanamycin
# ofloxacin
# levofloxacin
# cefoxitin
# imipenem

errors = list()
for ab in antibiotics:

    base_url = 'ftp://ftp.patricbrc.org/genomes/'
    extension = '.PATRIC.spgene.tab'
    genomes = set()
    """
    below lines were used for retrieving data for the january submission
    """
    # genome_file_dir = 'data_files/genome_ids'
    # base_out = 'data_files/sp_genes/' + ab + '/'
    """
    New lines for the retrieval pattern for the whole tree (april submission)
    """
    genome_file_dir = 'patric_cli'
    base_out = 'patric_cli/sp_genes/' + ab + '/'

    for file in os.listdir(genome_file_dir):
        if ab in file:
            print(file)
            # df = pd.read_csv(os.path.join(genome_file_dir, file), sep=',', dtype=str)
            # genomes = genomes.union(set(df['Genome ID']))
            df = pd.read_csv(os.path.join(genome_file_dir, file), sep='\t', dtype=str)
            genomes = genomes.union(set(df['genome_drug.genome_id']))
    # print('df')
    for genome in genomes:
        try:
            print(genome)
            fp = base_url + genome + '/' + genome + extension
            outfile = base_out + genome + '_spgene.tab'
            wget.download(fp, outfile)
        except:
            errors.append(genome)
    print('done ' + str(ab))
print('genomes with errors:')
print(errors)