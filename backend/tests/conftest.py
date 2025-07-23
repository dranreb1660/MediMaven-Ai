import pytest
import asyncio
from typing import AsyncGenerator
from unittest.mock import Mock, patch, AsyncMock
import os
import sys

# Set test environment
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///test.db"
os.environ["ENABLE_MONITORING"] = "false"
os.environ["JWT_SECRET"] = "test-secret-key"
os.environ["TESTING"] = "true"

# Test data
TEST_MEDICAL_QUERIES = [
    "What are symptoms of diabetes?",
    "Recommended dosage for metformin",
    "Side effects of ACE inhibitors",
    "Latest guidelines for hypertension treatment"
]

TEST_CITATIONS = [
    {"id": "1", "source": "Mayo Clinic", "url": "https://mayo.edu/diabetes", "rank": 1},
    {"id": "2", "source": "NIH", "url": "https://nih.gov/metformin", "rank": 2},
    {"id": "3", "source": "FDA", "url": "https://fda.gov/drugs", "rank": 3}
]

# Fixtures that don't require app import
@pytest.fixture
def mock_medimaven():
    """Mock MediMaven service"""
    mock = AsyncMock()
    mock.answer_rag = AsyncMock(return_value={
        "answer": "Based on medical literature, symptoms include...",
        "citations": TEST_CITATIONS[:1],
        "latency_s": 1.5
    })
    mock.prepare_stream = AsyncMock(return_value=([], "test prompt"))
    mock.stream_generator = AsyncMock()
    mock.rewrite_followup = AsyncMock(return_value="Rewritten query")
    mock.clean_text = Mock(side_effect=lambda x: x.strip() if x else "")
    return mock

@pytest.fixture
def mock_user():
    """Mock authenticated user"""
    return {"sub": "test-user-123", "email": "doctor@medimaven.ai"}

@pytest.fixture
def mock_db_session():
    """Mock database session"""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session

# Helper to create async generator
async def async_generator(items):
    """Helper to create async generator for testing"""
    for item in items:
        yield item

# Client fixture that patches database during import
@pytest.fixture
async def client():
    """Create test client with mocked database"""
    # Patch database before importing
    with patch('sqlalchemy.ext.asyncio.create_async_engine') as mock_engine:
        mock_engine.return_value = Mock()
        
        # Now safe to import
        from httpx import AsyncClient, ASGITransport
        from backend.app.main import app
        
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

# Simple mock app for basic tests
@pytest.fixture
def mock_app():
    """Create a mock FastAPI app"""
    from fastapi import FastAPI
    app = FastAPI()
    
    @app.get("/health")
    def health():
        return {"status": "ok", "bot_initialized": True}
    
    @app.post("/chat")
    async def chat(request: dict):
        return {
            "answer": "Test response",
            "citations": [],
            "latency": 0.1,
            "conversation_id": "test-123",
            "messages": []
        }
    
    return app
