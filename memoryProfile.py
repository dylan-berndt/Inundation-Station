import numpy as np

lines = open("output.txt").readlines()

allCount = []
count = {}
for line in lines:
    if "|" in line:
        allCount.append(count)
        count = {}
        continue

    left, right = line.find("["), line.find("]")
    shape = line[left + 1:right]

    if shape not in count:
        count[shape] = 0

    count[shape] += 1


count1 = allCount[0]
keys = list(count1.keys())
for count in allCount:
    for key in count:
        if key not in keys:
            continue
        if count[key] <= count1[key]:
            continue
        print(count[key], ":", key)
    print("\n\n")

