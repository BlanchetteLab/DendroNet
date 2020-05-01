import os
import argparse

dpf_list = [1.0, 0.1, 0.01, 0.001, 0.0001]

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Patric experiments tuning DPF')
    # parser.add_argument('--antibiotic', type=str, default='methicillin', metavar='A',
    #                     help='Antibiotic of interest')
    # args = parser.parse_args()
    #
    # ab = str(args.antibiotic)
    # tree_file = 'patric_application/patric_tree_storage/' + ab
    # label_file = ab + '_firmicutes_samples.csv'

    for dpf in dpf_list:
        output_dir = str('fungi_search' + str(dpf).replace('.', ''))

        command = str('python3 fungi_experiment.py --delta-penalty-factor ' + str(dpf) + ' --output-dir ' + output_dir)
        os.system(command)


