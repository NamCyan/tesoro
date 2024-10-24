import numpy as np
import os
import json
from datasets import load_dataset
import sklearn.metrics as metric
from transformers import AutoTokenizer
from tqdm import tqdm

src = "/cm/archive/namlh35/SATD/results/my-dataset-0.0.1/codebert-base-{}-concat-kfolds_comment"
data_src = "/cm/archive/namlh35/SATD/My_dataset/BATCH1/kfolds_0.0.1"
td_types = ["DEFECT", "DESIGN", "IMPLEMENTATION", "TEST", "DOCUMENTATION", "NONSATD"]
contexts = ["full_code", "code_context_2", "code_context_10", "code_context_20"]

acc_scores = []
f1_scores = []

summary = {}
for project in tqdm(os.listdir(src.format(contexts[0]))):
    if not os.path.isdir(os.path.join(src.format(contexts[0]), project)) or "maldonado_projects" in project:
            continue
    # print(project)
    predictions = {}
    for context in contexts:
        

        y_pred = []
        if not os.path.exists(os.path.join(src.format(context), project, f"{project}_predict_results.txt")):
            continue

        
    
        with open(os.path.join(src.format(context), project, f"{project}_predict_results.txt"), "r") as f:
            for line in f.readlines()[1:]: # skip header
                task_id, prediction = [x.strip() for x in line.replace("\n", "").split("\t")]
                # prediction = td_types.index(prediction)
                y_pred.append(prediction) 
        predictions[context] = y_pred

    vote_prediction = [[predictions[context][i] for context in predictions] for i in range(len(predictions[contexts[0]]))]
    vote_prediction = [max(set(x), key=x.count) for x in vote_prediction]
    y_pred = np.array([td_types.index(x) for x in vote_prediction])

    gt_data = load_dataset("json", data_files = os.path.join(data_src, f"{project}.json"), split="train")

    # tokenizer = AutoTokenizer.from_pretrained(os.path.join(src, project))
    # gt_data = gt_data.filter(lambda x: len(tokenizer.encode(x["cleancode"])) <= 512)
    y_true = np.array([td_types.index(x) for x in gt_data["classification"]])

    acc = metric.accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = metric.f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    summary[project] = {"acc": acc, "f1": f1}
    
    acc_scores.append(acc)
    f1_scores.append(f1)


summary["avg"] = {"accuracy": np.mean(acc_scores), "f1_score": np.mean(f1_scores)}

# print(json.dumps(summary, indent=4))
with open(os.path.join(src.format(contexts[0]), "ensemble.json"), "w") as f: 
    json.dump(summary, f, indent=4)