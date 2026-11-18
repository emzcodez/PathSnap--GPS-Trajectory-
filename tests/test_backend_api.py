"""Unit Tests for Backend FastAPI Service

Author: Prateek (Joint API Ownership with Priyansh)
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure root is in path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
grandparent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

try:
    from PathSnap.backend.main import app
except ModuleNotFoundError:
    from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert "PathSnap" in data["system"]


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_list_trips():
    res = client.get("/api/trips?partition=student_test")
    assert res.status_code == 200
    data = res.json()
    assert "trips" in data
    assert len(data["trips"]) > 0


def test_predict_route_schema():
    # Test on trip 00000000 (from student_test)
    payload = {"trip_id": "00000000", "partition": "student_test"}
    res = client.post("/api/predict_route", json=payload)
    assert res.status_code == 200
    data = res.json()

    # Validate strict Contract D adherence
    assert data["trip_id"] == "00000000"
    assert "gps_points" in data
    assert "predicted_arc_ids" in data
    assert "predicted_route_geometry" in data
    assert "metrics" in data
    assert "processing_time_ms" in data
    assert data["metrics"]["relaxed_route_continuity"] > 0.0

