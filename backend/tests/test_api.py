import pytest
from unittest.mock import patch, Mock, AsyncMock
from backend.tests.conftest import TEST_MEDICAL_QUERIES, TEST_CITATIONS

# Tests that don't require app import
class TestSchemas:
    """Test schemas without importing the full app"""
    
    def test_chat_request_schema(self):
        """Test request schema validation"""
        from backend.app.schemas import ChatRequest
        
        # Valid request
        req = ChatRequest(query="What are symptoms?")
        assert req.query == "What are symptoms?"
        
        # With optional fields
        req2 = ChatRequest(
            query="Test",
            conversation_id="123",
            history=[{"user": "Q", "assistant": "A"}]
        )
        assert req2.conversation_id == "123"
        assert len(req2.history) == 1

    def test_response_schema(self):
        """Test response schema"""
        from backend.app.schemas import ChatResponse, Citation, ConversationMessage
        
        response = ChatResponse(
            answer="Test answer",
            citations=[Citation(id="1", source="Test", url="http://test.com", rank=1)],
            latency=1.5,
            conversation_id="123",
            messages=[ConversationMessage(user="Q", assistant="A")]
        )
        
        assert response.answer == "Test answer"
        assert len(response.citations) == 1

# Tests that mock the database and import app
@pytest.mark.asyncio
class TestAPI:
    """API tests with mocked dependencies"""
    
    @patch('sqlalchemy.ext.asyncio.create_async_engine')
    async def test_health_endpoint(self, mock_engine):
        """Test health endpoint"""
        mock_engine.return_value = Mock()
        
        from httpx import AsyncClient, ASGITransport
        from backend.app.main import app
        
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"

    @patch('sqlalchemy.ext.asyncio.create_async_engine')
    @patch('backend.app.main.bot')
    async def test_chat_validation(self, mock_bot, mock_engine):
        """Test chat input validation"""
        mock_engine.return_value = Mock()
        
        from httpx import AsyncClient, ASGITransport
        from backend.app.main import app
        
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Empty query
            response = await client.post("/chat", json={"query": ""})
            assert response.status_code == 400
            
            # Query too long
            response = await client.post("/chat", json={"query": "x" * 10001})
            assert response.status_code == 400

    @patch('sqlalchemy.ext.asyncio.create_async_engine')
    @patch('backend.app.main.bot')
    @patch('backend.app.main.response_cache')
    async def test_successful_chat(self, mock_cache, mock_bot, mock_engine):
        """Test successful chat request"""
        mock_engine.return_value = Mock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()
        
        mock_bot.answer_rag = AsyncMock(return_value={
            "answer": "Diabetes symptoms include increased thirst...",
            "citations": TEST_CITATIONS[:1],
            "latency_s": 1.5
        })
        mock_bot.rewrite_followup = AsyncMock(return_value="Original query")
        
        from httpx import AsyncClient, ASGITransport
        from backend.app.main import app
        
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/chat", json={
                "query": TEST_MEDICAL_QUERIES[0],
                "history": []
            })
            
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "citations" in data
            assert len(data["citations"]) > 0

# Simplified mock-based tests
@pytest.mark.asyncio
class TestMockBased:
    """Tests using mocks without importing the app"""
    
    async def test_medimaven_mock(self, mock_medimaven):
        """Test MediMaven mock behavior"""
        result = await mock_medimaven.answer_rag("Test query")
        assert "answer" in result
        assert result["latency_s"] == 1.5
        
    async def test_streaming_mock(self, mock_medimaven):
        """Test streaming mock"""
        docs, prompt = await mock_medimaven.prepare_stream("Test query")
        assert isinstance(docs, list)
        assert isinstance(prompt, str)
