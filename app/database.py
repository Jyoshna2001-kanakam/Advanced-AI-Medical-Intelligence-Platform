"""
database.py
-----------
SQLAlchemy database setup. Defaults to SQLite for zero-config local/dev use;
set DATABASE_URL to a PostgreSQL/MySQL DSN for production deployment
(e.g. postgresql://user:pass@host:5432/medical_ai) without changing any
other code.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./medical_ai.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
