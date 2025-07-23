# MediMaven Test Suite

Industry-grade testing setup for the MediMaven medical RAG assistant.

## Quick Start

```bash
# Install dependencies
cd frontend && npm install
cd ../backend && pip install pytest pytest-asyncio pytest-cov httpx

# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e

# Watch mode (frontend only)
npm run test:watch
```

## Test Structure

```
MediMaven/
├── frontend/
│   └── tests/
│       ├── setup.ts         # Test configuration
│       ├── utils.tsx        # Shared test utilities
│       ├── unit/           # Component & hook tests
│       └── e2e/            # Cypress E2E tests
├── backend/
│   └── tests/
│       ├── conftest.py      # Pytest configuration
│       ├── test_api.py      # API endpoint tests
│       └── test_services.py # Service layer tests
└── test.sh                  # Test runner script
```

## Testing Philosophy

1. **Simple**: Minimal setup, clear structure
2. **Fast**: Unit tests run in <5s, full suite in <30s
3. **Reliable**: No flaky tests, proper mocking
4. **Focused**: Test medical RAG features, not frameworks

## Key Test Scenarios

### Frontend
- Chat interface interactions
- Medical query submission
- Citation display
- Streaming responses
- Error handling

### Backend
- RAG pipeline accuracy
- API validation
- Caching behavior
- Conversation management
- Medical content retrieval

### E2E
- Complete medical query flow
- Multi-turn conversations
- Citation verification
- Error recovery

## CI/CD

Tests run automatically on:
- Every push to `main` or `develop`
- All pull requests
- Security scans on main branch

## Coverage Goals

- Frontend: 80%+ coverage
- Backend: 85%+ coverage
- Focus on critical paths (medical queries, RAG pipeline)
