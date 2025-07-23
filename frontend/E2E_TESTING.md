# E2E Testing Setup for MediMaven

This document explains how to run End-to-End (E2E) tests for the MediMaven frontend.

## Overview

The E2E tests use Cypress and require both a frontend server and a mock backend server to be running. The mock backend simulates the actual API endpoints that the frontend interacts with.

## Prerequisites

- Node.js and npm installed
- Python with FastAPI and uvicorn installed (`pip install fastapi uvicorn`)
- Frontend dependencies installed (`npm install`)
- Frontend built (`npm run build`)

## Quick Start

### Option 1: Using the automated script (Recommended)

```bash
# Start both servers and run tests
npm run e2e:test

# Or start servers manually for interactive testing
npm run e2e:servers
# Then in another terminal:
npm run cypress:open
```

### Option 2: Manual setup

1. **Start the mock backend:**
```bash
python mock_backend.py
# Backend will run on http://localhost:8000
```

2. **Start the frontend server:**
```bash
npm run build
npm run preview -- --host 0.0.0.0 --port 5173
# Frontend will run on http://localhost:5173
```

3. **Run Cypress tests:**
```bash
# Headless mode
npm run cypress:run

# Interactive mode
npm run cypress:open
```

## Mock Backend

The mock backend (`mock_backend.py`) provides:

- `/health` endpoint for server health checks
- `/chat` endpoint for regular chat requests
- `/chat/stream` endpoint for streaming chat responses (Server-Sent Events)

### Mock Response Format

The mock backend returns responses in this format:
```json
{
  "answer": "This is a mock response for testing purposes. Your query was: [user_query]",
  "conversation_id": "test-conversation-id", 
  "citations": [{"id": "1", "title": "Mock Medical Document", "content": "Test citation"}],
  "latency_s": 0.5
}
```

## Test Files

- `tests/e2e/chat.cy.ts` - Basic chat functionality tests
- `tests/e2e/happyFlow.cy.ts` - Complete user workflow tests  
- `tests/e2e/medical-rag.cy.ts` - Medical RAG-specific functionality tests

## Configuration

The Cypress configuration is in `cypress.config.ts`:

- `baseUrl`: http://localhost:5173 (frontend)
- `apiUrl`: http://localhost:8000 (mock backend)
- Default timeouts are set for reliable testing

## Troubleshooting

### Frontend server not responding
- Ensure the frontend is built: `npm run build`
- Check that port 5173 is not already in use
- Wait a few seconds after starting the server before running tests

### Backend server issues
- Ensure FastAPI and uvicorn are installed: `pip install fastapi uvicorn`
- Check that port 8000 is not already in use
- Verify the backend is responding: `curl http://localhost:8000/health`

### Test failures
- Check that both servers are running and accessible
- Look at Cypress screenshots in `cypress/screenshots/` for visual debugging
- Increase timeouts if tests are failing due to slow responses

## CI/CD Integration

In CI environments, the GitHub Actions workflow automatically:
1. Builds the frontend
2. Starts the mock backend server
3. Starts the frontend preview server  
4. Runs the Cypress tests
5. Uploads screenshots and videos on failure

The same mock backend code is used both locally and in CI for consistency.
