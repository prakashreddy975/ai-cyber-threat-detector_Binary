from sqlalchemy.orm import Session
from .models import Event, Alert

def create_event(db: Session, *, features: dict, result: dict, actual_label: str | None = None) -> Event:
    ev = Event(
        features=features,
        prediction=int(result["prediction"]),
        threat_score=float(result["threat_score"]),
        risk_level=str(result["risk_level"]),
        threat_detected=bool(result["threat_detected"]),
        threshold=float(result["threshold"]),
        latency_ms=float(result["latency"]),
        actual_label=actual_label,
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)
    return ev

def create_alert(db: Session, *, event_id: int | None, result: dict) -> Alert:
    al = Alert(
        event_id=event_id,
        prediction=int(result["prediction"]),
        threat_score=float(result["threat_score"]),
        risk_level=str(result["risk_level"]),
        threshold=float(result["threshold"]),
        latency_ms=float(result["latency"]),
        note=None,
    )
    db.add(al)
    db.commit()
    db.refresh(al)
    return al

def list_alerts(db: Session, limit: int = 50, risk: str | None = None):
    q = db.query(Alert).order_by(Alert.created_at.desc())
    if risk:
        q = q.filter(Alert.risk_level == risk.upper())
    return q.limit(limit).all()

def list_events(db: Session, limit: int = 100):
    return db.query(Event).order_by(Event.created_at.desc()).limit(limit).all()
