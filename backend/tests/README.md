# Backend Testing Guide

## Running Tests

### Basic Test Run
```bash
cd backend
pytest -v
```

### Run with Coverage
```bash
pytest --cov=app --cov=services --cov-report=term-missing
```

### Run Specific Tests
```bash
# Only API tests
pytest tests/test_api.py -v

# Only service tests
pytest tests/test_services.py -v

# Skip slow/integration tests
pytest -m "not slow" -v
```

## Test Categories

1. **API Tests** (`test_api.py`)
   - Health endpoint
   - Chat endpoints (sync and streaming)
   - Input validation
   - Error handling

2. **Service Tests** (`test_services.py`)
   - MediMaven RAG pipeline
   - Caching functionality
   - Text processing
   - Error recovery

## Common Issues

### "bot not initialized" errors
The tests mock the MediMaven bot. If you see initialization errors, ensure:
- The mock_medimaven fixture is applied to your test
- You're not running against a real database

### Async warnings
All cache operations are async. Make sure to:
- Use `await` for cache.get() and cache.set()
- Mark test functions with `@pytest.mark.asyncio`

### Skipped tests
Some tests are skipped by default as they require:
- Vector database setup
- GPU/model availability
- External services

To run ALL tests (including slow ones):
```bash
pytest --run-slow
```

## Environment Variables

For integration tests, you may need:
```bash
export PYTHONPATH=/path/to/MediMaven
export DATABASE_URL=sqlite:///test.db
export ENABLE_MONITORING=false
```
