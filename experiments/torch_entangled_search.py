import os
import argparse


#dpfs = [0.5, 0.75, 1.0, 1.25]
#mutation_probs = [1.0]
#mrs = [0.4, 0.6, 0.8, 1.0]

# dpfs = [0.75]
# mutation_probs = [1.0]
# mrs = [0.6]

# [0.0, 0.25, 0.5, 1.0, 2.0]
# [0.0, 0.1, 1.0, 1.0]



# dpfs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
mutation_probs = [1.0]
# mrs = [0.0, 0.125, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0]

dpfs = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 1.0]
mrs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Delta search')
    parser.add_argument('--epochs', type=int, default=8000, metavar='S')
    parser.add_argument('--depth', type=int, default=4, metavar='D')
    parser.add_argument('--lr', type=float, default=0.01, metavar='L')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    counter = 0
    for mr in mrs:
        for mp in mutation_probs:
            for dpf in dpfs:
                output_dir = str('torch_gpu_test' + str(args.depth) + '_' + str(args.epochs) + '_' + str(args.seed) + '_'+
                # output_dir = str('torch_gpu_test' + str(args.depth) + '_' + str(args.epochs) + '_' +
                                 str(args.lr) + '_' + str(counter)).replace('.', '')
                counter += 1

                command = str(
                    'python3 entangled_sim_pytorch.py --delta-penalty-factor ' + str(dpf) + ' --mr ' + str(mr) +
                    ' --mp ' + str(mp) + ' --output-dir ' + output_dir + ' --lr ' + str(args.lr) +
                    ' --epochs ' + str(args.epochs) + ' --depth ' + str(args.depth) + ' --seed ' + str(args.seed))
                    # ' --epochs ' + str(args.epochs) + ' --depth ' + str(args.depth))
                os.system(command)
