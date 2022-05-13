import os


out_dir = './output'
titles = ['Dense', 'No-pool', 'DiffPool', 'Top-K', 'SAGpool', 'MinCutPool']
cols = ['dense', 'flat', 'diff_pool', 'top_k_pool', 'sag_pool', 'mincut_pool', 'mincut_noorthogonal']
rows = ['easy', 'hard', 'Mutagenicity', 'PROTEINS']

with open('./graph_classification.csv', 'w') as f_out:
    f_out.write('Dataset,' + ','.join(titles) + '\n')
    for dataset in rows:
        res = []
        for method in cols:
            try:
                f = open(os.path.join(out_dir, dataset, method + '.txt'))
                res.append(f.readline().split()[2])
                f.close()
            except:
                raise ValueError
        f_out.write(dataset + ',' + ','.join(res) + '\n')
