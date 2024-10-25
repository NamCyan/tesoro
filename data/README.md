# Data for Experiments

- [`detector_data`](detector_data.json): Data for training SATD Detectors.

*Format:*
```json
{
    "comment": "comment text",
    "classification": "techincal debt types (DESIGN | IMPLEMENTATION | DEFECT | DOCUMENTATION | TEST | WITHOUT_CLASSIFICATION)"
}
```

- [`tesoro_as_extra_data`](tesoro_as_extra_data.json): Data in the same format as [Maldonaldo](https://github.com/maldonado/tse.satd.data). Use $\text{Tesoro}_{comment}$ as additional source to answer RQ1.

*Format:*
```json
{
    "comment": "comment text",
    "classification": "techincal debt types (DESIGN | IMPLEMENTATION | DEFECT | DOCUMENTATION | TEST | WITHOUT_CLASSIFICATION)",
    "projectname":"repository name" // unsused when training
}
```

- [`10-folds maldonado62k`](kfolds/maldonado62k/): 10-folds/projects training and validation of Maldonado62K dataset. Use to answer RQ1. Data format is same as `tesoro_as_extra_data`.

- [`10-folds tesoro_comment`](kfolds/tesoro_comment/): 10-folds training and validation of  $\text{Tesoro}_{comment}$ dataset. Use to answer RQ2. 

*Format:*
```json
{
    "id": "function id in the dataset",
    "comment_id": "comment id of the function",
    "comment": "comment text",
    "classification": "techincal debt types (DESIGN | IMPLEMENTATION | DEFECT | DOCUMENTATION | TEST | NONSATD)",
    "code": "full fucntion context",
    "code_context_2": "2 lines code context",
    "code_context_10": "10 lines code context",
    "code_context_20": "20 lines code context",
    "repo": "Repository that contains this source" // unsused when training
}
```

- [`10-folds tesoro_code`](kfolds/tesoro_code/): 10-folds training and validation of  $\text{Tesoro}_{code}$ dataset. Use to answer RQ3. 

*Format:*
```json
{
    "id": "function id in the dataset",
    "original_code": "raw function",
    "code": "original code without comment",
    "cleancode": "normalized version of code (lowercase, remove newline \n)",
    "label": "binary list corresponding to 4 TD types (DESIGN, IMPLEMENATION, DEFECT, TEST)",
    "repo": "Repository that contains this source" // unsused when training
}
```