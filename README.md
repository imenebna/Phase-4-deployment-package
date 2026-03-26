This repository contains the deployment package for my fine tuned BART summarization project.



GitHub: https://github.com/imenebna/Phase-4-deployment-package

Model repo: https://huggingface.co/Imenebna/bart-phase4-model

Live app: https://huggingface.co/spaces/Imenebna/bart-phase4-app



\## What is included here

This GitHub repository contains the application and deployment code:

\- `app.py`

\- `requirements.txt`

\- `Dockerfile`

\- `.streamlit/config.toml`

\- deployment notes and submission text



\## Deployment structure

This project uses Streamlit as the application framework.



Because the trained model file is very large, the project is split across three platforms:



1\. \*\*GitHub\*\* for the application code and deployment files

&#x20;  https://github.com/imenebna/Phase-4-deployment-package



2\. \*\*Hugging Face Model Hub\*\* for the fine tuned BART model files

&#x20;  https://huggingface.co/Imenebna/bart-phase4-model



3\. \*\*Hugging Face Spaces\*\* for the live deployed Streamlit application

&#x20;  https://huggingface.co/spaces/Imenebna/bart-phase4-app



\## Why the model is not stored in GitHub

The file `model.safetensors` is about 1.55 GB, which exceeds GitHub's standard file size limit. For that reason, the code is stored in GitHub, while the trained model is hosted separately on Hugging Face Model Hub.



\## Phase 4 objective

In this phase, I deployed my fine tuned BART summarization model through a Streamlit application so a user can input a long text and receive an abstractive summary.

