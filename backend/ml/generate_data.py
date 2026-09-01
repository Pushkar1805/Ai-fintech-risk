from pathlib import Path
import numpy as np, pandas as pd
SEED=42; rng=np.random.default_rng(SEED); BASE=Path(__file__).resolve().parents[1]/'data'; BASE.mkdir(exist_ok=True)
N=10000
income=rng.lognormal(np.log(45000),.45,N).clip(12000,300000)
expenses=(income*rng.uniform(.35,.8,N)).clip(5000,None)
inflow=income*rng.normal(1.05,.12,N).clip(.7,1.5); outflow=expenses*rng.normal(1.05,.12,N).clip(.7,1.5)
txn_count=rng.poisson(65,N)+5; avg_txn=(outflow/txn_count*rng.uniform(.7,1.3,N)).clip(100,15000)
savings=(income*rng.uniform(.2,3.5,N)).clip(500,1000000); failed=rng.poisson(.8,N); late=rng.poisson(.7,N); age=rng.integers(3,61,N)
vol=rng.uniform(.05,.65,N); weekend=rng.uniform(.1,.45,N); night=rng.uniform(.02,.25,N); new=rng.binomial(1,.12,N); logins=rng.poisson(.6,N); loc=rng.binomial(1,.08,N)
credit_raw=(failed*5+late*7+vol*35+(expenses/income)*25+(age<9)*8+(inflow<outflow*1.02)*18+rng.normal(0,8,N)).clip(0,100)
fraud_raw=(new*28+np.minimum(logins,5)*8+loc*25+(avg_txn>12000)*18+(night>.2)*12+rng.normal(0,7,N)).clip(0,100)
df=pd.DataFrame({'customer_id':[f'C{i:05d}' for i in range(N)],'monthly_income':income.round(2),'monthly_expenses':expenses.round(2),'average_monthly_inflow':inflow.round(2),'average_monthly_outflow':outflow.round(2),'transaction_count':txn_count,'average_transaction':avg_txn.round(2),'savings_balance':savings.round(2),'failed_payments':failed,'late_payments':late,'account_age_months':age,'spending_volatility':vol.round(4),'weekend_spending_ratio':weekend.round(4),'night_transaction_ratio':night.round(4),'new_device':new,'failed_logins':logins,'location_change':loc,'high_credit_risk':(credit_raw>55).astype(int),'fraud_label':(fraud_raw>55).astype(int)})
df.to_csv(BASE/'customers.csv',index=False)
rows=[]
for i in range(N):
 k=max(1,min(20,txn_count[i]//8)); amounts=np.abs(rng.normal(avg_txn[i],max(50,avg_txn[i]*vol[i]),k))
 for j,a in enumerate(amounts): rows.append((f'C{i:05d}',j+1,round(float(a),2),int(new[i] and j==k-1),int(loc[i] and j==k-1),int(a>12000)))
pd.DataFrame(rows,columns=['customer_id','transaction_id','amount','new_device_event','location_change_event','high_value_event']).to_csv(BASE/'transactions.csv',index=False)
print(f'Generated {N} synthetic customers and {len(rows)} transactions')
