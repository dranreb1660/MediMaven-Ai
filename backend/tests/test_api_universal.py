import pytest
from unittest.mock import patch, Mock, AsyncMock
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test data
TEST_MEDICAL_QUERIES = [
    "What are symptoms of diabetes?",
    "Recommended dosage for metformin",
    "Side effects of ACE inhibitors",
    "Latest guidelines for hypertension treatment"
]

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

@pytest.mark.skipif(
    os.environ.get("CI") == "true" and not os.environ.get("SKIP_INTEGRATION"),
    reason="Skip integration tests in CI unless explicitly enabled"
)
@pytest.mark.asyncio
class TestAPIIntegration:
    """API integration tests - skipped in CI by default"""
    
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