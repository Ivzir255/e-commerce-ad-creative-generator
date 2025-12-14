# E-Commerce Ad Creative Generator with MLOps Pipeline

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your credentials

# 3. Run locally
python src/data/data_ingestion.py
python src/models/train.py
uvicorn src.api.main:app --reload

# 4. Deploy
docker build -t ad-generator-api:latest -f docker/Dockerfile.api .
kubectl apply -f k8s/
```

## Project Structure

See the artifact "Project Directory Structure" for complete layout.

## Documentation

- [Architecture](docs/architecture.md)
- [Deployment Guide](docs/deployment.md)
- [API Reference](docs/api.md)
- [Runbook](docs/runbook.md)

## Next Steps

1. Copy all code files from the Claude artifacts
2. Configure your cloud credentials
3. Prepare your product data
4. Follow the deployment guide

For detailed setup instructions, refer to the complete README artifact.
