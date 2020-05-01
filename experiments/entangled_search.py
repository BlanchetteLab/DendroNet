import os
import argparse


dpfs = [0.5, 0.75, 1.0, 1.25]
mutation_probs = [1.0]
mrs = [0.4, 0.6, 0.8, 1.0]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Delta search')
    parser.add_argument('--num-steps', type=int, default=10000, metavar='S')
    parser.add_argument('--depth', type=int, default=9, metavar='D')
    parser.add_argument('--lr', type=float, default=0.001, metavar='L')
    args = parser.parse_args()
    counter = 0
    for mr in mrs:
        for mp in mutation_probs:
            for dpf in dpfs:
                output_dir = str('entangled_search_l1' + str(args.depth) + '_' + str(args.num_steps) + '_' +
                                 str(args.lr) + '_' + str(counter)).replace('.', '')
                counter += 1

                command = str(
                    'python3 entangled_delta_model.py --delta-penalty-factor ' + str(dpf) + ' --mr ' + str(mr) +
                    ' --mp ' + str(mp) + ' --output-dir ' + output_dir + ' --lr ' + str(args.lr) +
                    ' --num-steps ' + str(args.num_steps) + ' --depth ' + str(args.depth))
                os.system(command)
