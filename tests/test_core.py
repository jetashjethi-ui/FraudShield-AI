"""
FraudShield AI — Unit Tests
Tests core functionality without requiring trained model artifacts.
"""

import pytest
import sys
import os
import ast
import json
import importlib.util

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════════════
#  Test 1: All Python files parse without syntax errors
# ══════════════════════════════════════════════════════════════════════
class TestCodeQuality:
    """Verify all Python files are syntactically valid."""

    def _get_py_files(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        py_files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ('.git', '__pycache__', 'catboost_info', '.streamlit')]
            for f in filenames:
                if f.endswith('.py'):
                    py_files.append(os.path.join(dirpath, f))
        return py_files

    def test_all_python_files_parse(self):
        """Every .py file must parse without SyntaxError."""
        errors = []
        for path in self._get_py_files():
            try:
                with open(path, encoding='utf-8') as f:
                    ast.parse(f.read())
            except SyntaxError as e:
                errors.append(f"{path}: {e}")
        assert errors == [], f"Syntax errors found:\n" + "\n".join(errors)

    def test_minimum_file_count(self):
        """Project should have at least 15 Python files."""
        assert len(self._get_py_files()) >= 15

    def test_dashboard_line_count(self):
        """Dashboard should be substantial (2000+ lines)."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dashboard = os.path.join(root, 'dashboard.py')
        with open(dashboard, encoding='utf-8') as f:
            lines = sum(1 for _ in f)
        assert lines >= 2000, f"Dashboard has {lines} lines, expected 2000+"


# ══════════════════════════════════════════════════════════════════════
#  Test 2: Module imports work
# ══════════════════════════════════════════════════════════════════════
class TestImports:
    """Verify core modules can be imported."""

    def test_import_numpy(self):
        import numpy as np
        assert hasattr(np, 'array')

    def test_import_pandas(self):
        import pandas as pd
        assert hasattr(pd, 'DataFrame')

    def test_import_sklearn(self):
        from sklearn.model_selection import StratifiedKFold
        assert StratifiedKFold is not None

    def test_import_fastapi(self):
        from fastapi import FastAPI
        assert FastAPI is not None


# ══════════════════════════════════════════════════════════════════════
#  Test 3: API endpoints
# ══════════════════════════════════════════════════════════════════════
class TestAPI:
    """Test API endpoints without running the server."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from fastapi.testclient import TestClient
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from api import app
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert "models_loaded" in data

    def test_landing_page(self):
        response = self.client.get("/")
        assert response.status_code == 200
        assert "FraudShield" in response.text

    def test_docs_endpoint(self):
        response = self.client.get("/docs")
        assert response.status_code == 200

    def test_predict_endpoint(self):
        payload = {
            "TransactionAmt": 500.0,
            "ProductCD": "C",
            "hour_of_day": 14,
            "is_weekend": 0,
            "card4": "visa",
            "card6": "debit",
            "P_emaildomain": "gmail.com",
            "card1": 5000,
            "addr1": 300,
        }
        # Try /predict first, fall back to /score
        response = self.client.post("/predict", json=payload)
        if response.status_code == 422:
            response = self.client.post("/score", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "risk_score" in data
        assert "risk_category" in data
        assert 0 <= data["risk_score"] <= 100

    def test_predict_high_risk(self):
        """High-risk transaction should score higher."""
        payload = {
            "TransactionAmt": 45000.0,
            "ProductCD": "W",
            "hour_of_day": 3,
            "is_weekend": 0,
            "card4": "discover",
            "card6": "credit",
            "P_emaildomain": "protonmail.com",
            "card1": 9999,
            "addr1": 100,
        }
        response = self.client.post("/predict", json=payload)
        data = response.json()
        assert data["risk_score"] > 40, "High-risk transaction should score above 40"

    def test_model_info_endpoint(self):
        response = self.client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "architecture" in data
        assert data["architecture"]["detection_layers"] == 25


# ══════════════════════════════════════════════════════════════════════
#  Test 4: Feature engineering functions
# ══════════════════════════════════════════════════════════════════════
class TestFeatureEngineering:
    """Test that feature engineering modules have expected functions."""

    def test_feature_engine_exists(self):
        spec = importlib.util.find_spec("src.feature_engine")
        assert spec is not None, "src.feature_engine module not found"

    def test_graph_engine_exists(self):
        spec = importlib.util.find_spec("src.graph_engine")
        assert spec is not None, "src.graph_engine module not found"

    def test_risk_scorer_exists(self):
        spec = importlib.util.find_spec("src.risk_scorer")
        assert spec is not None, "src.risk_scorer module not found"

    def test_adversarial_validation_exists(self):
        spec = importlib.util.find_spec("src.adversarial_validation")
        assert spec is not None, "src.adversarial_validation module not found"

    def test_threshold_optimizer_exists(self):
        spec = importlib.util.find_spec("src.threshold_optimizer")
        assert spec is not None, "src.threshold_optimizer module not found"


# ══════════════════════════════════════════════════════════════════════
#  Test 5: Project structure
# ══════════════════════════════════════════════════════════════════════
class TestProjectStructure:
    """Verify required files and directories exist."""

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @pytest.mark.parametrize("filename", [
        "main.py", "dashboard.py", "api.py",
        "requirements.txt", "Dockerfile", "README.md",
        "MODEL_CARD.md", "DOCUMENTATION.md", "LICENSE",
        "render.yaml",
    ])
    def test_required_files(self, filename):
        assert os.path.exists(os.path.join(self.ROOT, filename)), f"{filename} missing"

    @pytest.mark.parametrize("dirname", [
        "src", "landing", "mobile",
    ])
    def test_required_dirs(self, dirname):
        assert os.path.isdir(os.path.join(self.ROOT, dirname)), f"{dirname}/ missing"

    def test_mobile_has_pwa_files(self):
        mobile = os.path.join(self.ROOT, "mobile")
        for f in ["index.html", "app.js", "style.css", "manifest.json", "sw.js"]:
            assert os.path.exists(os.path.join(mobile, f)), f"mobile/{f} missing"

    def test_landing_has_index(self):
        assert os.path.exists(os.path.join(self.ROOT, "landing", "index.html"))


# ══════════════════════════════════════════════════════════════════════
#  Test 6: Threshold optimizer logic
# ══════════════════════════════════════════════════════════════════════
class TestThresholdOptimizer:
    """Test threshold optimization module exists and is importable."""

    def test_module_importable(self):
        from src.threshold_optimizer import run_threshold_optimization
        assert callable(run_threshold_optimization)

    def test_function_signature(self):
        import inspect
        from src.threshold_optimizer import run_threshold_optimization
        sig = inspect.signature(run_threshold_optimization)
        assert len(sig.parameters) >= 1, "Function should accept at least 1 parameter"
