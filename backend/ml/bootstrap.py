from pathlib import Path
import subprocess, sys
BASE=Path(__file__).resolve().parents[1]
if not (BASE/'data/customers.csv').exists(): subprocess.check_call([sys.executable,str(BASE/'ml/generate_data.py')])
if not (BASE/'trained_models/credit_model.joblib').exists(): subprocess.check_call([sys.executable,str(BASE/'ml/train_models.py')])
