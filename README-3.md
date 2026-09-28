# Basketball Shot Probability Modeling

A Python project that estimates the probability of a basketball shot going in from shot-level data. I built it as an analyst internship technical assessment, with a focus on preparing the data, evaluating probability estimates, and explaining the model's behavior.

## Project overview

- **Training data:** 425,719 labeled, non-fouled shots
- **Output:** 213,977 predicted shot probabilities in `submission.csv`
- **Validation metric:** log loss, where lower is better
- **Constant-probability baseline:** 0.689627 validation log loss
- **Model:** 0.628205 validation log loss

The model improved validation log loss by approximately 0.0614 compared with the baseline. I used permutation importance to inspect which inputs had the largest effect on predictions; shot distance was the strongest feature in that analysis.

## Approach

1. Load the provided training and testing data.
2. Prepare shot-level features and fit a probability model in Python.
3. Compare validation log loss with a constant-probability baseline.
4. Inspect permutation importance to help interpret the model.
5. Generate the probabilities in `submission.csv` and document the methods and results in `project_writeup.pdf`.

## Repository files

| File | Description |
| --- | --- |
| `project_code.py` | Data preparation, modeling, evaluation, and prediction code |
| `project_writeup.pdf` | Project methods, findings, and limitations |
| `training.csv.gz` | Labeled shot data supplied for the assessment |
| `testing.csv.gz` | Unlabeled shot data supplied for the assessment |
| `submission.csv` | Generated shot probabilities |

## Running the project

Use a Python environment with the packages imported in `project_code.py`. From the repository directory, run:

```bash
python project_code.py
```

The script expects the supplied data files in the same directory. See `project_writeup.pdf` for the modeling decisions and evaluation details.

## Scope

This project uses basketball shot data for a technical assessment. Its reported validation result describes performance on the assessment's validation split; it does not establish performance on future seasons or other datasets.
