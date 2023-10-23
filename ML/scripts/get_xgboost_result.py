from collections import defaultdict
from prettytable import PrettyTable
from statistics import quantiles, mean
from itertools import chain

with open("xgboost_output/log") as f:
    lines = f.readlines()

def analysis(samples):
    return max(samples), min(samples), mean(samples), 
    
lines = [line.strip() for line in lines]
indices = [index for index, line in enumerate(lines) if '=' in line]
step_size = indices[1] - indices[0] - 2
rmse_getter = lambda x: float(x[x.rfind(':') + 1:])
gt_getter = lambda x: float(x[:x.rfind("1:")-1])
columns = [x.strip().split('=')[0] for x in lines[indices[0]].split(',')][:-1]

data = {
    'rmse':defaultdict(list), 
    'wt-wise':defaultdict(lambda :defaultdict(list)), 
    'sim-wise':defaultdict(lambda :defaultdict(list))}
tags = []
for index in indices:
    header = [x.strip().split('=') for x in lines[index].split(',')]
    tag = '.'.join([x[1] for x in header if x[0] != 'fold'])
    fold = [x[1] for x in header if x[0] == 'fold'][0]
    if tag not in tags:
        tags.append(tag)
    data['rmse'][tag].append(rmse_getter(lines[index + step_size]))

    with open(f"xgboost_output/{tag}.{fold}.pred") as file:
        y_pred = [ float(x.strip()) for x in file.readlines()]

    with open(f"data/fold{fold}/test.scale") as file:
        y_true = [ gt_getter(x.strip()) for x in file.readlines() ]

    assert len(y_pred) == len(y_true)
    assert len(y_pred) % 80 == 0

    g = defaultdict(lambda :defaultdict(list))
    for index, (pred, gt) in enumerate(zip(y_pred, y_true)):
        wt_id = index % 80
        gid = index // 80
        data['wt-wise'][tag][fold].append(100*abs(pred-gt)/gt if gt > 0 else 0)
        g[gid]['pred'].append(pred)
        g[gid]['gt'].append(gt)

    for gid in g:
        pred = mean(g[gid]['pred'])
        gt = mean(g[gid]['gt'])
        data['sim-wise'][tag][fold].append(100*abs(pred-gt)/gt if gt > 0 else 0)

for tag in tags:
    data['sim-wise'][tag] = quantiles(list(chain.from_iterable(list(data['sim-wise'][tag].values()))), n=101)
    data['wt-wise'][tag] = quantiles(list(chain.from_iterable(list(data['wt-wise'][tag].values()))), n=101)

t = PrettyTable()
t.field_names  = ["Params", "Quantity", "Q1", "Q5", "Q30", "Q50", "Q70", "Q95", "Q99"]
for tag in tags:
    param=f"({', '.join(tag.split('.'))})"
    t.add_row([param] + [" "]*(len(t.field_names)-1))
    for quantity in ['WT-wise', 'Sim-wise']:
        values = data[quantity.lower()][tag]
        values = [ values[i] for i in [0, 4, 29, 49, 69, 94, 98] ]
        values = [f"{x:.4e}" for x in values]
        t.add_row([" ", quantity+" (%)"] + values, divider=quantity=='Sim-wise')

with open("xgboost.result.txt", "w") as f:
    f.write(str(t))
