import csv
import math
import numpy as np

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

num_of_trains = int(len(data) * 0.8)
indices = np.random.permutation(len(data))

x_train = []
y_train = []

for index in indices[:num_of_trains]:
    sample = data[index]

    angle = sample['angle']
    vel = sample['vel']
    x, y, p = sample['data']
    for i in range(x.shape[0]):
        x_train.append(np.append(np.append(x, y), np.array([angle, vel, x[i], y[i]])))
        y_train.append(p[i])

x_test = []
y_test = []

for index in indices[num_of_trains:]:
    sample = data[index]

    angle = sample['angle']
    vel = sample['vel']
    x, y, p = sample['data']
    for i in range(x.shape[0]):
        x_test.append(np.append(np.append(x, y), np.array([angle, vel, x[i], y[i]])))
        y_test.append(p[i])

with open("./train.dat", "w") as file:
    for i in range(len(x_train)):
        features = ' '.join([f"{ind + 1:d}:{x:.4f}" for ind, x in enumerate(x_train[i])])
        if i + 1 != len(x_train):
            file.write(f"{y_train[i]:.4f} {features}\n")
        else:
            file.write(f"{y_train[i]:.4f} {features}")

with open("./test.dat", "w") as file:
    for i in range(len(x_test)):
        features = ' '.join([f"{ind + 1:d}:{x:.4f}" for ind, x in enumerate(x_test[i])])
        if i + 1 != len(x_test):
            file.write(f"{y_test[i]:.4f} {features}\n")
        else:
            file.write(f"{y_test[i]:.4f} {features}")
