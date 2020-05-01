import os
import numpy as np

bias_files = ['bias_output_0', 'bias_output_1', 'bias_output_2', 'bias_output_3', 'bias_output_4']
fungi_files = ['fungi_output_0', 'fungi_output_1', 'fungi_output_2', 'fungi_output_3', 'fungi_output_4']
trophic_levels = ['S', 'N', 'B', 'OB', 'HB', 'MP', 'P', 'ECM', 'E', 'L', 'C', 'T']
log_file = 'auc_log_file'
output_file = 'parsed_combined_results.txt'

lines = list()
bias_scores = dict()
tree_scores = dict()
simple_scores = dict()

for lvl in trophic_levels:
    bias_scores[lvl] = list()
    tree_scores[lvl] = list()
    simple_scores[lvl] = list()


for dir in bias_files:
    f = os.path.join(dir, log_file)
    curr_file = open(f, 'r').readlines()
    for line in curr_file:
        line = line.split()
        lvl = line[0][:-1]
        score = float(line[-1])
        bias_scores[lvl].append(score)

for dir in fungi_files:
    f = os.path.join(dir, log_file)
    curr_file = open(f, 'r').readlines()
    for line in curr_file:
        line = line.split()
        lvl = line[0][:-1]
        tree_score = float(line[3])
        simple_score = float(line[-1])
        tree_scores[lvl].append(tree_score)
        simple_scores[lvl].append(simple_score)


print('done')
for lvl in trophic_levels:
    tm = round(np.mean(tree_scores[lvl]), 3)
    tstd = round(np.std(tree_scores[lvl]), 3)

    sm = round(np.mean(simple_scores[lvl]), 3)
    sstd = round(np.std(simple_scores[lvl]), 3)

    bm = round(np.mean(bias_scores[lvl]), 3)
    bstd = round(np.std(bias_scores[lvl]), 3)
    line = str(lvl + '  tree= ' + str(tm) + ' +/- ' + str(tstd) + '  bias= '
               + str(bm) + ' +/- ' + str(bstd) + '  simple= ' + str(sm) + ' +/- ' + str(sstd) + '\n\n')
    lines.append(line)

    out = open(output_file, 'w+')
    out.writelines(lines)
