import os
import subprocess
# os.system('p3-all-drugs')

drug_list = list(str(os.popen("p3-all-drugs").read()).split('\n'))[1:-1]

for drug in drug_list:
    command = str("p3-echo -t antibiotic " + drug + " | p3-get-drug-genomes")
print(drug_list)