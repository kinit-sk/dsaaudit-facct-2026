# DSA Study

This repository documents the analysis workflow in three files:

- dsa_study.py
- dsa_study.ipynb
- dsa_annotations.ipynb

## dsa_study.py

A command-line script that runs the full analysis pipeline end-to-end and writes the resulting tables to disk. It expects the input CSV to be present in the current working directory.

### Run

```bash
python dsa_study.py
```

## dsa_study.ipynb

An exploratory notebook that loads the generated tables and produces figures and summary views. It is intended for interactive analysis and visualization.

### Run

Open the notebook in Jupyter or VS Code and execute the cells from top to bottom.

## dsa_annotations.ipynb

An annotation notebook that loads the annotation data and produces confusion matrices for both annotators and the model used in the study.

### Run

Open the notebook in Jupyter or VS Code and execute the cells from top to bottom.
