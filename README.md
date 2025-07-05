# HousingRegression---Assignment
# Housing Regression ML Pipeline

This project is part of ML Ops Assignment 1. It demonstrates how to build, automate, and manage a classical machine learning pipeline using Git, GitHub Actions, and Conda environments.

## Problem Statement

Predict house prices using the Boston Housing dataset. Compare performance of at least 3 regression models using MSE and R². Automate model training using GitHub Actions CI pipeline.

## Models Used

- Linear Regression / Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor

## Branch Structure

- `main` — Final merged version with README
- `reg` — Contains base models and evaluation
- `hyper` — Adds hyperparameter tuning (min 3 per model)

## Workflow Automation

GitHub Actions is configured to:

- Trigger on every push to `main`, `reg`, `hyper`
- Set up Python environment
- Install dependencies from `requirements.txt`
- Run `regression.py` and check model performance

See `.github/workflows/ci.yml` for full workflow.

## Repo Structure

HousingRegression/
├── .github/workflows/ci.yml
├── utils.py
├── regression.py
├── requirements.txt
└── README.md

## Evaluation Metrics

Each model is evaluated using:
- Mean Squared Error (MSE)
- R² Score

## Final Report

The final report `p22cs201_A1.pdf` contains:
- GitHub repo link
- Steps followed
- Results table
- Best parameters (tuned)
- Summary and learnings
