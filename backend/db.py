import os
import asyncio
from dotenv import load_dotenv
from sqlalchemy import Column, Integer, Text, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

load_dotenv()
# URL hack: inject asyncpg and strip SSL params to avoid driver errors
URL = (
    os.getenv("NEON_DATABASE_URL")
    .replace("postgresql://", "postgresql+asyncpg://")
    .split("?")[0]
)
engine = create_async_engine(URL, connect_args={"ssl": True})
Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    embedding = Column(Vector(768))


async def init():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    print("Database initialization complete.")


if __name__ == "__main__":
    asyncio.run(init())
