import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow importing from app
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from sqlalchemy.future import select
from app.db.models.user import UserModel
from app.db.models.doctor import DoctorModel
from app.core.auth import get_password_hash
from app.config.settings import settings

# Updated to include first and last names
DOCTORS = [
    ("dr.who@example.com", "John", "Smith", "Cardiology"),
    ("dr.house@example.com", "Gregory", "House", "Diagnostic Medicine"),
    ("dr.chen@example.com", "Mei", "Chen", "Neurology"),
]

async def main() -> None:
    print("Connecting to database at:", settings.database_url)
    engine = create_async_engine(str(settings.database_url))
    async_session = sessionmaker(engine, class_=AsyncSession)

    async with async_session() as db:
        for email, first_name, last_name, specialty in DOCTORS:
            # Check if doctor already exists to avoid duplicates
            # Using text() to properly wrap the SQL query
            existing_user = await db.execute(
                select(UserModel).where(UserModel.email == email)
            )
            user_result = existing_user.first()

            if user_result is not None:
                print(f"Doctor with email {email} already exists. Skipping.")
                continue

            pwd_hash = get_password_hash("TestPassword1!")
            user = UserModel(email=email, password_hash=pwd_hash, role="doctor")
            user.doctor_profile = DoctorModel(
                first_name=first_name,
                last_name=last_name,
                specialty=specialty
            )
            db.add(user)
            print(f"Added doctor: {first_name} {last_name} ({email}), specialty: {specialty}")

        await db.commit()
        print("Doctors successfully added to the database.")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
