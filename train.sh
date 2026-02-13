#!/usr/bin/env bash
source venv/bin/activate
python -m jupyter nbconvert --to script train.ipynb 
python train.py ; rm train.py