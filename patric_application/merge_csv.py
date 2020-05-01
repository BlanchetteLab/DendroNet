import os
import pandas as pd


# csv_files = ['data_files/0_bac_PATRIC_sp_gene.csv', 'data_files/clostridium_PATRIC_sp_gene.csv']
csv_files = ['data_files/0_bac_PATRIC_sp_gene.csv', 'data_files/clostridium_PATRIC_sp_gene.csv']
outfile = '0_bac_clostridium_merge_sp_gene'

outfile += '.csv'

df_list = list()

for f in csv_files:
    df_list.append(pd.read_csv(f))

# now we need to drop whatever extra columns some file may have
df = pd.concat(df_list, ignore_index=True)
df.to_csv(path_or_buf=os.path.join('data_files', outfile), index=False)
print('done')