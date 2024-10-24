import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

src = "/cm/archive/namlh35/backup/AI4Code/SATD/data_with_code_v3/"
full_data = []
functions = set()
with open(os.path.join(src, "satd.json"), "r") as f:
    for dp in f:
        dp = json.loads(dp)
        full_data.append(dp)
        functions.add(dp["function"].strip())
    
# with open(os.path.join(src, "nonsatd.json"), "r") as f:
#     for dp in f:
#         dp = json.loads(dp)
#         full_data.append(dp)
#         functions.add(dp["function"].strip())

print("Number of comments in phase 1:",len(full_data))


# print("Number of functions:", len(functions))

satd_types = ["DESIGN", "DOCUMENTATION", "DEFECT", "IMPLEMENTATION", "TEST"]
data_by_type = {}

for satd_type in satd_types:
    with open(os.path.join(src, f"satd_{satd_type}.json"), "r") as f:
        data_by_type[satd_type] = {json.loads(dp)['id']:json.loads(dp) for dp in f.readlines()}

for satd_type in satd_types:
    assert len(data_by_type[satd_type].keys()) == len(data_by_type[satd_type].keys())


overlap_type = {satd_type: [] for satd_type in satd_types}
for type1 in satd_types:
    for type2 in satd_types:
        if type1 == type2:
            overlap_type[type1].append(1.)
        else:
            overlap_type[type1].append(len(set(data_by_type[type1].keys()).intersection(set(data_by_type[type2].keys())))/len(set(data_by_type[type1].keys()).union(set(data_by_type[type2].keys()))))

overlap_matrix = np.array([overlap_type[satd_type] for satd_type in satd_types])
print(overlap_matrix)

mask = np.zeros_like(overlap_matrix)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(overlap_matrix, mask=mask, vmax=.3, square=True,  cmap="crest", xticklabels=satd_types, yticklabels=satd_types,  cbar=False)
    # plt.show()
# plt.tight()
# plt.xticks(rotation=45) 
plt.savefig("confusion.pdf", bbox_inches='tight')

