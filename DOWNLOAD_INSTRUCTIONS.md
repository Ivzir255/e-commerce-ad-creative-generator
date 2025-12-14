# Download Instructions

## Manual Download Steps

For each artifact in the Claude conversation, follow these steps:

### 1. Data Ingestion (data_ingestion)
- Click download button
- Save as: `src/data/data_ingestion.py`

### 2. Model Training (model_training)
- Click download button
- Save as: `src/models/train.py`

### 3. FastAPI Service (fastapi_service)
- Click download button
- Save as: `src/api/main.py`

### 4. Airflow DAGs (airflow_dags)
- Click download button
- Split file into two parts at the "# airflow/dags/training_pipeline_dag.py" comment
- Save first part as: `airflow/dags/data_ingestion_dag.py`
- Save second part as: `airflow/dags/training_pipeline_dag.py`

### 5. Docker & Kubernetes (docker_k8s_config)
- Click download button
- Split file by "---" separators
- Save Dockerfile part as: `docker/Dockerfile.api`
- Save each K8s manifest as separate files in `k8s/` directory

### 6. GitHub Actions (github_actions)
- Click download button
- Save as: `.github/workflows/ci-cd.yml`

### 7. Monitoring Config (monitoring_config)
- Click download button
- Split file by "---" separators
- Save prometheus.yml in: `monitoring/prometheus/`
- Save alerts in: `monitoring/prometheus/alerts/`
- Save docker-compose in: `monitoring/`

### 8. Grafana Dashboard (grafana_dashboard)
- Click download button
- Save as: `monitoring/grafana/dashboards/ml-metrics.json`

### 9. Test Suite (test_suite)
- Click download button
- Split into separate test files
- Save each in `tests/` directory

### 10. Requirements (requirements_file)
- Already created by this script as `requirements.txt`

### 11. README (readme_file)
- Click download button
- Replace the basic `README.md` created by this script

## Alternative: Use Git

If you prefer, you can:
1. Create a new GitHub repository
2. Copy files manually from artifacts
3. Commit and push

## Verification

After downloading all files, verify:

```bash
# Check Python syntax
python -m py_compile src/**/*.py

# Check YAML syntax
yamllint k8s/*.yaml

# Run tests
pytest tests/ -v
```

## Next Steps

1. Configure `.env` file with your credentials
2. Install dependencies: `pip install -r requirements.txt`
3. Follow README.md for deployment instructions
