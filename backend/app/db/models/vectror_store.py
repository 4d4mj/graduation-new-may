# backend/app/db/models/vector_store.py
import uuid
from sqlalchemy import (
    Column,
    String,
    ForeignKey,
    Index,
    JSON,
    TEXT,
    UniqueConstraint,
    PrimaryKeyConstraint,
    Integer,  # Added Integer for SERIAL
    MetaData,  # Import MetaData for naming convention
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.db.base import Base  # Use your existing Base

# --- Import the Vector type ---
from pgvector.sqlalchemy import Vector
# ----------------------------

# --- Optional: Naming convention for Alembic ---
# Helps Alembic generate constraint names consistently
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table)s",
    "pk": "pk_%(table_name)s",
}
metadata_obj = MetaData(naming_convention=convention)
# If your Base already has metadata with a convention, you might not need this.
# If Base = declarative_base(), you can pass metadata: Base = declarative_base(metadata=metadata_obj)
# For now, let's assume Base is simple. We can adjust if needed.
# ---------------------------------------------


class CollectionStore(Base):
    __tablename__ = "langchain_pg_collection"
    # __table_args__ = ( # Constraints defined below using convention if possible
    #     PrimaryKeyConstraint('uuid', name='langchain_pg_collection_pkey'), # Use convention: pk_langchain_pg_collection
    #     UniqueConstraint('name', name='langchain_pg_collection_name_key'), # Use convention: uq_langchain_pg_collection_name
    # )

    uuid = Column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )  # Set primary_key=True here
    name = Column(String, unique=True)  # Set unique=True here
    cmetadata = Column(JSON)

    embeddings = relationship("EmbeddingStore", back_populates="collection")


class EmbeddingStore(Base):
    __tablename__ = "langchain_pg_embedding"
    # __table_args__ = ( # Define indexes separately
    #      PrimaryKeyConstraint('id', name='langchain_pg_embedding_pkey'),
    #      UniqueConstraint('uuid', name='langchain_pg_embedding_uuid_key'),
    # )

    # Let's use SERIAL for ID as previously decided, SQLAlchemy represents this as Integer+autoincrement
    id = Column(
        Integer, primary_key=True, autoincrement=True
    )  # Use Integer, primary_key=True, autoincrement=True for SERIAL
    uuid = Column(
        UUID(as_uuid=True), unique=True, nullable=False, default=uuid.uuid4
    )  # Keep unique UUID
    collection_id = Column(
        UUID(as_uuid=True),
        ForeignKey("langchain_pg_collection.uuid", ondelete="CASCADE"),
    )

    # --- Use the Vector type ---
    embedding = Column(Vector(768))  # Replace 768 with your dimension if different
    # ---------------------------

    document = Column(TEXT)
    cmetadata = Column(JSON)
    custom_id = Column(String)

    collection = relationship("CollectionStore", back_populates="embeddings")

    # Define indexes explicitly for clarity and potential naming convention use
    __mapper_args__ = {"eager_defaults": True}  # Helps with default values like UUID
    Index("embedding_collection_id_idx", collection_id)
    # Index("embedding_custom_id_idx", custom_id) # Uncomment if needed
