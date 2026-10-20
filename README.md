# Heart Disease MLOps: end-to-end project

> **Educational use only.** This model is not a medical device and must not be used to diagnose, treat, or make decisions about real patients.

## What this repository demonstrates
- Downloading and cleaning the UCI Heart Disease Cleveland dataset
- EDA plots and summary tables
- Logistic Regression and Random Forest with stratified 5-fold CV and a held-out test set
- MLflow parameters, metrics, fitted model and plots
- Reproducible preprocessing pipeline and saved model
- FastAPI `/predict`, `/health`, `/metrics`, request logs
- Pytest, Ruff and GitHub Actions CI
- Docker packaging and Minikube deployment

## Architecture
```text
UCI dataset -> download/clean -> EDA -> train + CV + MLflow -> joblib model
                                                           |
Client -> NodePort/Service -> FastAPI /predict -> preprocessing + classifier
                             | /health /metrics /logs
GitHub Actions -> lint + tests + training artifact
```

## Prerequisites
Install Python 3.11, Git, Docker Desktop (or Docker Engine), kubectl, and Minikube. Enable virtualization if required by your system. Verify:
```bash
python --version
docker --version
kubectl version --client
minikube version
```
Commands below are shown for Windows PowerShell where Windows-specific activation differs. On macOS/Linux, use `python3` and `source .venv/bin/activate`.

### Automated setup and deployment on Rocky Linux 9

From a regular (non-root) user account on the Rocky Linux 9 VM, clone this repository and run the bootstrap script:
```bash
git clone <YOUR_REPOSITORY_URL>
cd 2025ae05570-mlops
bash scripts/install_rocky9_minikube.sh
```
The script checks existing RPM packages and commands, installs only missing prerequisites, verifies downloaded `kubectl` and Minikube checksums, sets up the project Python environment, trains the model if it is missing, runs the CI lint/tests, then builds and deploys the image to Minikube. All output is logged to `~/rocky9-minikube-setup-<timestamp>.log`. It may need internet access to install packages and download the dataset.

If the script adds your account to the `docker` group, log out and back in, then rerun it. This is required for Docker/Minikube to work without running the cluster as root. Defaults request 2 CPUs and 4096 MB for Minikube; override them if needed:
```bash
MINIKUBE_CPUS=4 MINIKUBE_MEMORY=6g bash scripts/install_rocky9_minikube.sh
```
On success, the script prints the Minikube NodePort URLs for `/docs`, `/health`, and `/metrics`. Access from another machine may also require allowing TCP port 30080 through the VM's network/firewall; the script does not change firewall policy automatically.

## 1. Set up the environment
```powershell
git clone <YOUR_REPOSITORY_URL>
cd heart-disease-mlops
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. You can also use `.\.venv\Scripts\python.exe` without activation.

## 2. Download the dataset
```powershell
python scripts/download_data.py
```
The script fetches the official UCI archive and writes `data/raw/heart.csv`. Internet access is needed for this step. The Cleveland source uses `?` for missing values and a target code 0–4; this project maps target 0 to class 0 and values 1–4 to class 1.

## 3. Run EDA
```powershell
python scripts/make_eda.py
```
Review `reports/eda_summary.txt` and `reports/figures/` (histograms, class balance, correlation heatmap). The preprocessing pipeline imputes missing values during training to avoid fitting imputers on test data.

## 4. Train and compare models
```powershell
python -m src.train
```
The training script uses a stratified 80/20 train/test split and stratified 5-fold cross-validation on the training partition. It tunes Logistic Regression C and Random Forest tree count/depth, selects the final model by held-out ROC-AUC, and writes:
- `models/heart_model.joblib`
- `models/metadata.json`
- `reports/training_summary.json`
- MLflow tracking database in `mlflow.db`
- MLflow runs in `mlruns/`
- confusion matrices and ROC plots in `reports/figures/`

Metrics are computed when you run it; do not copy example or guessed metrics into your report. For this small dataset, results can vary by library version and split. Explore experiments with:
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```
Open `http://127.0.0.1:5000`.

## 5. Test and lint
```powershell
ruff check src scripts tests
pytest -q
```
The API test suite covers health/root endpoints and request validation. Run tests after training too. CI runs tests and lint before training.

## 6. Run the API locally
Ensure training completed and `models/heart_model.joblib` exists:
```powershell
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
Visit `http://127.0.0.1:8000/docs` or `http://127.0.0.1:8000/health`.
Example request (synthetic demonstration values only):
```powershell
$body = @{
 age=55; sex=1; cp=2; trestbps=130; chol=240; fbs=0; restecg=1
 thalach=150; exang=0; oldpeak=1.0; slope=2; ca=0; thal=3
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body $body
```
The response includes the predicted class, model probability and confidence. These are model outputs, not clinical risk estimates. `/metrics` exposes Prometheus metrics; application logs go to the terminal.

## 7. Build and test the Docker image
Stop the local Uvicorn process. Build from the project root after training:
```powershell
docker build -t heart-disease-api:1.0 .
docker run --rm -p 8000:8000 heart-disease-api:1.0
```
In another terminal, test `http://localhost:8000/health` and submit the sample request above. Capture a screenshot of successful build and test for your report.

## 8. Deploy to Minikube
Start a local Kubernetes cluster:
```powershell
minikube start --driver=docker
kubectl cluster-info
```
Build the image directly into Minikube's image store:
```powershell
minikube image build -t heart-disease-api:1.0 .
```
Apply manifests and wait for rollout:
```powershell
kubectl apply -f deployment/kubernetes.yaml
kubectl rollout status deployment/heart-api
kubectl get pods,svc
```
Open the service:
```powershell
minikube service heart-api --url
```
Use the printed URL (often a localhost tunnel URL) in your test. If the terminal needs to stay open for the tunnel, leave it running. Alternatively use:
```powershell
kubectl port-forward service/heart-api 8000:8000
```
Then test `http://127.0.0.1:8000/health`, `/docs`, `/metrics`, and POST `/predict`. Check logs:
```powershell
kubectl logs deployment/heart-api
kubectl describe pod -l app=heart-api
```
If the pod fails, inspect `kubectl get pods`, `kubectl describe pod <POD_NAME>`, and `kubectl logs <POD_NAME>`. Common cause: model artifact missing because training did not complete before image build.

## 9. Monitoring
Prometheus-compatible request and latency metrics are exposed at `/metrics`. Verify by opening that endpoint or running:
```powershell
(Invoke-WebRequest http://127.0.0.1:8000/metrics).Content
```
The supplied `monitoring/prometheus.yml` is a starting scrape configuration, not a complete Prometheus/Grafana installation. For a simple local demonstration, show `/metrics` and API logs. Optional next step: install Prometheus and Grafana in Minikube and configure a scrape target reachable from the Prometheus pod.

## 10. GitHub and CI
Create a GitHub repository, then:
```powershell
git init
git add .
git commit -m "Build heart disease MLOps project"
git branch -M main
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```
Replace `<YOUR_REPOSITORY_URL>` with your repository's actual URL. The GitHub Actions workflow runs Ruff, Pytest and then downloads data, generates EDA, trains the models, and uploads artifacts. Visit the Actions tab and capture a successful workflow run. Do not claim CI passed until you see a green run.

## 11. Screenshots and submission checklist
Save genuine screenshots in `screenshots/`:
1. EDA plots and summary
2. MLflow UI showing both model runs and metrics
3. successful pytest/Ruff output
4. Docker build and running container
5. Minikube pods and service (`kubectl get pods,svc`)
6. `/docs`, successful `/predict`, `/metrics`
7. GitHub Actions run
8. optional short screen recording showing the pipeline

Update `reports/training_summary.json` and the report with metrics generated by your own run. Include your name, course, date, repository URL, and screenshots. A public API URL is not created automatically by Minikube: Minikube is a local cluster; provide local access instructions or a screen recording.

## Dataset and methodology notes
Dataset: UCI Machine Learning Repository, Heart Disease dataset (Cleveland subset), ID 45. The original target values 0–4 are converted to binary: 0 means absence in source coding; 1–4 means presence. The dataset is small and historically collected. Results are for coursework demonstration and should not be generalized to current clinical populations.
