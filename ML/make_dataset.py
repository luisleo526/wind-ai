import csv
import math
import numpy as np
from pathlib import Path

data = {}
with open("../data/file1.csv", newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row_id, row in enumerate(reader):
        if row_id > 0:
            exp_id, exp_angle, exp_vel = row
            data[exp_id] = dict(id=exp_id, angle=float(exp_angle) * math.pi / 180.0,
                                vel=float(exp_vel), data=[])

with open("../data/file2.csv", newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row_id, row in enumerate(reader):
        if row_id > 0:
            exp_id, dev_id, x, y, power = row
            data[exp_id]['data'].append(list(map(float, [x, y, power])))

data = list(data.values())
for exp in data:
    exp['data'] = np.transpose(np.array(exp['data']), (1, 0))


Path("./data").mkdir(parents=True, exist_ok=True)

for k in range(5):

    with open(f"./data/{k}.txt", "w") as file:
        for sample in data[k::5]:
            angle = sample['angle']
            vel = sample['vel'] 
            x, y, p = sample['data']
            for i in range(x.shape[0]):
                feature = [angle, vel, x[i], y[i] ] + x.tolist() + y.tolist()
                feature = ' '.join([f"{ind + 1:d}:{x:.4f}" for ind, x in enumerate(feature)])
                file.write(f"{p[i]:.4f} {feature}\n")