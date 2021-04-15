import os

# antibiotics = ['methicillin', 'cloramphenicol']
# dpfs = [0.1, 0.01]

# antibiotics = ['betalactam', 'penicillin', 'ceftriaxone', 'ciprofloxacin', 'erythromycin', 'rifampin', 'tetracycline']
# dpfs = [0.01, 0.001, 0.001, 0.001, 0.01, 0.01, 0.001]

antibiotics = ['rifampin']
dpfs = [0.001]

output_dir_base = 'test_'
tree_folder_base = 'patric_application/patric_tree_storage/'
label_file_base = '_firmicutes_samples.csv'

for ab, dpf in zip(antibiotics, dpfs):
    output_dir = output_dir_base + str(ab)
    tree_folder = tree_folder_base + str(ab)
    label_file = str(ab) + label_file_base
    command = str('python3 patric_run_baseline_test.py --output-dir ' + output_dir + ' --tree-folder ' + tree_folder
                   + ' --label-file ' + label_file + ' --delta-penalty-factor ' + str(dpf))
    os.system(command)