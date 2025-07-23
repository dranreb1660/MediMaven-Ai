"""Simple tests to verify test setup is working"""
import pytest

def test_basic_math():
    """Test that basic math works"""
    assert 2 + 2 == 4

def test_string_operations():
    """Test string operations"""
    assert "med" + "imaven" == "medimaven"

@pytest.mark.asyncio
async def test_async_function():
    """Test async function execution"""
    async def get_value():
        return 42
    
    result = await get_value()
    assert result == 42

def test_imports():
    """Test that key imports work"""
    try:
        from backend.services.caching import InHouseCache
        from backend.services.generate import Backend
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
