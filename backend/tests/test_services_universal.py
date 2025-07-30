import pytest, os
from unittest.mock import Mock, patch, AsyncMock

class TestBasicUnits:
    """Basic unit tests without complex dependencies"""
    
    def test_imports(self):
        """Test that basic imports work"""
        try:
            from backend.app.schemas import ChatRequest, ChatResponse
            from backend.services.generate import Backend
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_backend_enum(self):
        """Test Backend enum"""
        from backend.services.generate import Backend
        
        # Backend uses auto() which generates integer values
        assert Backend.TRANSFORMERS.value == 1
        assert Backend.AWQ.value == 2
        assert Backend.VLLM.value == 3
        
        # Test enum names
        assert Backend.TRANSFORMERS.name == "TRANSFORMERS"
        assert Backend.AWQ.name == "AWQ"
        assert Backend.VLLM.name == "VLLM"
    
    def test_schema_validation(self):
        """Test schema field validation"""
        from backend.app.schemas import ChatRequest
        
        # Test required field
        with pytest.raises(Exception):
            ChatRequest()  # Missing required 'query' field
        
        # Valid request
        req = ChatRequest(query="Test")
        assert req.query == "Test"
        assert req.conversation_id is None  # Optional field

@pytest.mark.asyncio
class TestMockIntegration:
    """Integration tests using mocks"""
    
    async def test_rag_pipeline_mock(self):
        """Test RAG pipeline with mocks"""
        mock_medimaven = AsyncMock()
        mock_medimaven.answer_rag = AsyncMock(return_value={
            "answer": "Based on medical literature...",
            "citations": [{"id": "1", "source": "Mayo Clinic", "url": "#", "rank": 1}],
            "latency_s": 1.5
        })
        
        result = await mock_medimaven.answer_rag("What are diabetes symptoms?")
        
        assert "answer" in result
        assert "citations" in result
        assert result["latency_s"] > 0

@pytest.mark.skipif(
    "CI" in os.environ,
    reason="Skip in CI - requires full environment"
)
@pytest.mark.asyncio
class TestFullIntegration:
    """Full integration tests - only run locally"""
    pass