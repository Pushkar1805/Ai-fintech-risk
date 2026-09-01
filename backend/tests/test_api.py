from fastapi.testclient import TestClient
from app.main import app
client=TestClient(app)
def payload(): return {'customer_id':'TEST01','monthly_income':45000,'monthly_expenses':22000,'average_monthly_inflow':52000,'average_monthly_outflow':30000,'transaction_count':85,'average_transaction':1250,'savings_balance':50000,'failed_payments':1,'late_payments':0,'account_age_months':18,'spending_volatility':.15,'weekend_spending_ratio':.2,'night_transaction_ratio':.05,'new_device':False,'failed_logins':0,'location_change':False,'kyc_status':'VERIFIED','requested_credit_limit':20000}
def test_health(): assert client.get('/health').status_code==200
def test_assessment():
 r=client.post('/application/assess',json=payload()); assert r.status_code==200; assert 0<=r.json()['overall_risk_score']<=100
def test_kyc_gate():
 p=payload(); p['kyc_status']='FAILED'; assert client.post('/application/assess',json=p).json()['decision']=='MANUAL_REVIEW'
