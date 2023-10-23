from collections import defaultdict

from prettytable import PrettyTable
from statistics import mean

with open("xgboost_output/log") as f:
    lines = f.readlines()
    
lines = [line.strip() for line in lines]
indices = [index for index, line in enumerate(lines) if '=' in line]
step_size = indices[1] - indices[0] - 2
rmse_getter = lambda x: float(x[x.rfind(':') + 1:])
columns = [x.strip().split('=')[0] for x in lines[indices[0]].split(',')][:-1]

data = defaultdict(list)
for index in indices:
    header = [x.strip().split('=') for x in lines[index].split(',')]
    tag = '.'.join([x[1] for x in header if x[0] != 'fold'])
    data[tag].append(rmse_getter(lines[index + step_size]))

for key in data:
    data[key] = mean(data[key])

t = PrettyTable()
t.field_names = columns + ['RMSE']
for key in data:
    t.add_row(key.split('.') + [f"{data[key]:.6f}"])
    
print(t)
