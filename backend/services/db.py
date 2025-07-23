from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.app import config

print("Initializing database connection...")
print(f"Using DATABASE_URL: {config.DATABASE_URL}")
engine = create_async_engine(
    config.DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,   # ← validate connection before each checkout
    pool_recycle=1800,    # ← recycle every 30 min (avoids idle timeout)
)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
