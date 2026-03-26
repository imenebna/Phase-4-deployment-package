# Phase 4 Deployment Package for the Fine Tuned BART Model

## What this package contains
- `app.py` for the Streamlit inference app
- `requirements.txt` for Python dependencies
- `Dockerfile` for container deployment
- `.streamlit/config.toml` for server configuration

## Model files you must place inside `model/`
Create a folder named `model` inside this package and put these files inside it:
- `config.json`
- `generation_config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `training_args.bin` if you want to keep the full project assets together

## Local test
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Docker test
```bash
docker build -t bart-summarizer .
docker run -p 8501:8501 bart-summarizer
```

## Recommended deployment flow for this phase
1. Put the final model files inside the `model/` folder.
2. Test the app locally.
3. Push the package to a Git repository.
4. Deploy with Docker on a VM or cloud service that has enough memory for inference.
5. Submit the running app link.

## Notes
- The app loads the fine tuned model from local files and performs abstractive summarization.
- The decoding parameters can be adjusted from the sidebar.
- If the selected model folder is missing required files, the app stops and shows the missing items.
