# momentum_final_projects

Create and activate conda environment:
```bash
conda create -y -n momentum python="3.8.8"
conda activate momentum
```

Install requirements
```bash
pip install -r requirements.txt && pip install -e .
```

For development install pre-commit hooks
```
pip install pre-commit
pre-commit install
```
