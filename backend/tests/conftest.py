import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test data that works everywhere
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

@pytest.fixture
def mock_user():
    """Mock authenticated user"""
    return {"sub": "test-user-123", "email": "doctor@medimaven.ai"}

# Only set test database if not already set
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///test.db"
if "JWT_SECRET" not in os.environ:
    os.environ["JWT_SECRET"] = "test-secret-key"
if "ENABLE_MONITORING" not in os.environ:
    os.environ["ENABLE_MONITORING"] = "false"