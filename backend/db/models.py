from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime,Text, JSON, func
from .database import Base

class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)

    features = Column(JSON, nullable=False)

    prediction = Column(Integer, nullable=False)
    threat_score = Column(Float, nullable=False)
    risk_level = Column(String(16), nullable=False)
    threat_detected = Column(Boolean, nullable=False)

    threshold = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=False)

    actual_label = Column(String(64), nullable=True)

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)

    event_id = Column(Integer, nullable=True, index=True)

    prediction = Column(Integer, nullable=False)
    threat_score = Column(Float, nullable=False)
    risk_level = Column(String(16), nullable=False)
    threshold = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=False)

    note = Column(String(255), nullable=True)


class RagChunk(Base):
    __tablename__ = "rag_chunks"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)

    doc_name = Column(String(255), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)

    # store retrieval score snapshot optionally
    last_score = Column(Float, nullable=True)
