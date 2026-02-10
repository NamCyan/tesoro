import numpy as np
import os
import json
from datasets import load_dataset
import sklearn.metrics
from transformers import AutoTokenizer

def F1Measure_multi(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            temp+=1
            continue
        temp+= (2*sum(np.logical_and(y_true[i], y_pred[i])))/ (sum(y_true[i])+sum(y_pred[i]))
    return temp/ y_true.shape[0]

src = "/cm/archive/namlh35/SATD/results/my-dataset-0.0.1/graphcodebert-kfold_comment-fullcode-identification"
data_src = "/cm/archive/namlh35/SATD/My_dataset/BATCH1/kfolds_0.0.1"

acc_scores = []
f1_scores = []

summary = {}
for project in os.listdir(src):
    if not os.path.isdir(os.path.join(src, project)) or "maldonado_projects" in project:
        continue
    if "multilabel" in src:
        y_pred = []
        if not os.path.exists(os.path.join(src, project, f"{project}_predict_results.txt")):
            continue
        with open(os.path.join(src, project, f"{project}_predict_results.txt"), "r") as f:
            for line in f.readlines()[1:]: # skip header
                task_id, prediction = line.replace("\n", "").split("\t")
                prediction = [int(x) for x in list(prediction.split())]
                y_pred.append(prediction) 
        y_pred = np.array(y_pred)
        gt_data = load_dataset("json", data_files = os.path.join(data_src, f"{project}.json"), split="train")

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(src, project))
        gt_data = gt_data.filter(lambda x: len(tokenizer.encode(x["cleancode"])) <= 512)
        y_true = np.array(gt_data["label"])

        emr = np.all(y_pred == y_true, axis=1).mean()
        f1 = F1Measure_multi(y_true=y_true, y_pred=y_pred)
        summary[project] = {"EMR": emr, "f1": f1}
        
        acc_scores.append(emr)
        f1_scores.append(f1)
    else:
        with open(os.path.join(src, project, f"{project}_predict_score.json"), "r") as f:
            scores = json.load(f)

        summary[project] = scores

        if "my_dataset" in project or "my-dataset" in project or "llama" in project:
            print(scores)
            continue
        
        acc_scores.append(scores["accuracy"])
        f1_scores.append(scores["macro_f1"])


summary["avg"] = {"accuracy": np.mean(acc_scores), "f1_score": np.mean(f1_scores)}

with open(os.path.join(src, "summary.json"), "w") as f: 
    json.dump(summary, f, indent=4)
