with open("xgboost.log") as f:
    lines = f.readlines()

filter_func = lambda x: "max_depth" in x or '[99]' in x
data = list(filter(filter_func, lines))
data=[x.strip() for x in data]

getter = lambda x: int(x.strip().split('=')[-1])
score_getter = lambda x: float(x.split('[99]')[-1].strip()[10:])
for i in range(len(data)//2):
    depth, bin = data[2*i].split(',')
    depth = getter(depth)
    bin = getter(bin)
    score = score_getter(data[2*i+1])
    print(f"{depth}, {bin:3d}, {score:.6f}")
