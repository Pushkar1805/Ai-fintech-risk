from datetime import datetime
from sqlalchemy import String, Float, Integer, Boolean, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column
from .db import Base
class Customer(Base):
 __tablename__='customers'; id:Mapped[int]=mapped_column(primary_key=True); customer_id:Mapped[str]=mapped_column(String(32),unique=True,index=True); monthly_income:Mapped[float]=mapped_column(Float); savings_balance:Mapped[float]=mapped_column(Float); kyc_status:Mapped[str]=mapped_column(String(24),default='VERIFIED'); created_at:Mapped[datetime]=mapped_column(DateTime,default=datetime.utcnow)
class Assessment(Base):
 __tablename__='assessments'; id:Mapped[int]=mapped_column(primary_key=True); customer_id:Mapped[str]=mapped_column(String(32),index=True); overall_risk_score:Mapped[float]=mapped_column(Float); behavioral_risk_score:Mapped[float]=mapped_column(Float); fraud_risk_score:Mapped[float]=mapped_column(Float); alternative_credit_score:Mapped[float]=mapped_column(Float); decision:Mapped[str]=mapped_column(String(24)); recommended_credit_limit:Mapped[float]=mapped_column(Float); risk_factors:Mapped[str]=mapped_column(Text); created_at:Mapped[datetime]=mapped_column(DateTime,default=datetime.utcnow)
