from pathlib import Path
import json, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from xgboost import XGBClassifier
BASE=Path(__file__).resolve().parents[1]; df=pd.read_csv(BASE/'data/customers.csv')
FEATURES=['monthly_income','monthly_expenses','average_monthly_inflow','average_monthly_outflow','transaction_count','average_transaction','savings_balance','failed_payments','late_payments','account_age_months','spending_volatility','weekend_spending_ratio','night_transaction_ratio','new_device','failed_logins','location_change']
def train_target(target, prefix):
 X=df[FEATURES]; y=df[target]; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
 models={'logistic':Pipeline([('imp',SimpleImputer()),('sc',StandardScaler()),('m',LogisticRegression(max_iter=1200,class_weight='balanced'))]),'random_forest':Pipeline([('imp',SimpleImputer()),('m',RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=3,class_weight='balanced',random_state=42,n_jobs=-1))]),'xgboost':Pipeline([('imp',SimpleImputer()),('m',XGBClassifier(n_estimators=300,max_depth=5,learning_rate=.05,subsample=.85,colsample_bytree=.85,eval_metric='logloss',random_state=42))])}
 metrics={}; best=None; bestauc=-1
 for name,m in models.items():
  m.fit(Xtr,ytr); p=m.predict(Xte); pr=m.predict_proba(Xte)[:,1]; auc=roc_auc_score(yte,pr)
  metrics[name]={'precision':round(float(precision_score(yte,p,zero_division=0)),4),'recall':round(float(recall_score(yte,p,zero_division=0)),4),'f1':round(float(f1_score(yte,p,zero_division=0)),4),'roc_auc':round(float(auc),4),'pr_auc':round(float(average_precision_score(yte,pr)),4),'confusion_matrix':confusion_matrix(yte,p).tolist()}
  if auc>bestauc: bestauc=auc; best=m; bestname=name
 joblib.dump({'model':best,'features':FEATURES,'name':bestname,'target':target},BASE/f'trained_models/{prefix}_model.joblib'); return metrics,bestname
credit_metrics,credit_best=train_target('high_credit_risk','credit'); fraud_metrics,fraud_best=train_target('fraud_label','fraud')
iso=IsolationForest(n_estimators=300,contamination=.08,random_state=42); iso.fit(df[FEATURES]); joblib.dump({'model':iso,'features':FEATURES},BASE/'trained_models/anomaly_model.joblib')
metrics={'credit':credit_metrics,'fraud':fraud_metrics,'selected':{'credit':credit_best,'fraud':fraud_best},'training_rows':len(df)}
(BASE/'trained_models/metrics.json').write_text(json.dumps(metrics,indent=2)); print('Credit:',credit_best,'Fraud:',fraud_best)
