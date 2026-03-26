# What to do to complete Phase 4

## 1. Put the trained model files into the package
Inside the deployment package, create a folder called `model` and copy these files into it:
- `config.json`
- `generation_config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `training_args.bin` optionally

## 2. Test locally
Run:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the local Streamlit page and make sure the model generates a summary.

## 3. Put the package on GitHub
Upload the full folder to a new repository.

## 4. Choose a deployment target
Recommended: a VM or Docker capable cloud host.

## 5. Deploy
### Option A. Docker on a VM
```bash
docker build -t bart-summarizer .
docker run -d -p 8501:8501 bart-summarizer
```
Then open `http://YOUR_EXTERNAL_IP:8501`

### Option B. Plain server run
```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```
Then open `http://YOUR_EXTERNAL_IP:8501`

## 6. Submit
Submit the live app link and paste the short report from `phase4_submission_text.txt`.
