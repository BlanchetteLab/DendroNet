import os

seeds = [0, 1, 2, 3, 4]
# seeds = [0, 1]
bias_names = ['bias_output_0', 'bias_output_1', 'bias_output_2', 'bias_output_3', 'bias_output_4']
fungi_names = ['fungi_output_0', 'fungi_output_1', 'fungi_output_2', 'fungi_output_3', 'fungi_output_4']

for i in range(len(seeds)):
    command = str('python3 fungi_experiment.py --num-steps ' + str(1000) + ' --seed ' +
                  str(seeds[i]) + ' --output-dir ' + str(fungi_names[i]))
    os.system(command)

    command = str('python3 bias_experiment.py --num-steps ' + str(500) + ' --seed ' +
                  str(seeds[i]) + ' --output-dir ' + str(bias_names[i]))
    os.system(command)
