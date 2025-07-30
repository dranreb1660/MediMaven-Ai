# MediMaven Backend Tests - Google Colab Guide

## Quick Start

Run these commands in Google Colab:

```python
# 1. Setup environment
!pip install aiosqlite pytest pytest-asyncio httpx

# 2. Navigate to backend
import os
os.chdir('/content/drive/Othercomputers/kyei_mac_m3_pro/MediMaven/backend')

# 3. Run setup script
!python tests/setup_colab.py

# 4. Run basic tests
!pytest tests/test_minimal.py -v

# 5. Run service tests (without full app)
!pytest tests/test_services.py::TestBasicUnits -v
!pytest tests/test_services.py::TestMockIntegration -v

# 6. Run API tests (schemas only)
!pytest tests/test_api.py::TestSchemas -v
```

## Test Categories

### 1. **Minimal Tests** (`test_minimal.py`)
- Basic smoke tests
- Schema validation
- Mock functionality
- No external dependencies

### 2. **Service Tests** (`test_services.py`)
- Component unit tests
- Mock-based integration
- Cache functionality
- No database required

### 3. **API Tests** (`test_api.py`)
- Schema validation
- Mock-based endpoint tests
- Some require full app setup

## Troubleshooting

### "InvalidRequestError: asyncio extension requires async driver"
This happens when SQLite is loaded without aiosqlite. Solution:
```python
!pip install aiosqlite
```

### "ModuleNotFoundError: No module named 'backend'"
Set the Python path:
```python
import sys
sys.path.insert(0, '/content/drive/Othercomputers/kyei_mac_m3_pro/MediMaven')
```

### Tests hang or timeout
Some tests try to load models or connect to services. Skip them:
```python
!pytest -v -k "not integration and not full"
```

## Running Specific Test Groups

```bash
# Only schema tests (always work)
pytest -k "schema" -v

# Only mock tests (no dependencies)
pytest -k "mock" -v

# Only basic unit tests
pytest tests/test_services.py::TestBasicUnits -v

# Skip slow tests
pytest -m "not slow" -v
```

## Expected Results

When running in Colab, you should see:
- `test_minimal.py`: 6-8 tests passing ✓
- `test_services.py::TestBasicUnits`: 3 tests passing ✓ (includes Backend enum test)
- `test_services.py::TestMockIntegration`: 2 tests passing ✓
- `test_api.py::TestSchemas`: 2 tests passing ✓
- `test_services.py::TestCaching`: 2 tests passing ✓

Total: ~15-17 tests should pass

Full integration tests may fail due to missing GPU/models - this is expected.

## Notes

- The AutoAWQ deprecation warning is suppressed in pytest.ini
- The Backend enum uses integer values (1, 2, 3) not strings
- All tests are designed to work without GPU or model files

## Coverage Report

To see test coverage:
```bash
pytest --cov=backend --cov-report=term-missing -v
```
