import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

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

# Updated to use single-character values for the `sex` field
DOCTORS = [
    ("dr.who@example.com", "John", "Smith", "Cardiology", "M", "1975-05-15", "123-456-7890"),
    ("dr.who2@example.com", "Alice", "Johnson", "Cardiology", "F", "1980-07-20", "123-456-7891"),
    ("dr.house@example.com", "Gregory", "House", "Diagnostic Medicine", "M", "1965-06-11", "123-456-7892"),
    ("dr.chen@example.com", "Mei", "Chen", "Neurology", "F", "1985-03-25", "123-456-7893"),
    ("dr.chen2@example.com", "Li", "Wang", "Neurology", "M", "1990-09-10", "123-456-7894"),
    ("dr.brown@example.com", "Sarah", "Brown", "Pediatrics", "F", "1978-12-05", "123-456-7895"),
    ("dr.jones@example.com", "Michael", "Jones", "Orthopedics", "M", "1982-11-15", "123-456-7896"),
    ("dr.taylor@example.com", "Emily", "Taylor", "Dermatology", "F", "1992-01-30", "123-456-7897"),
    ("dr.martin@example.com", "James", "Martin", "General Surgery", "M", "1970-04-18", "123-456-7898"),
    ("dr.clark@example.com", "Anna", "Clark", "Oncology", "F", "1988-08-22", "123-456-7899"),
    ("dr.lewis@example.com", "David", "Lewis", "Radiology", "M", "1983-02-14", "123-456-7800"),
    ("dr.walker@example.com", "Sophia", "Walker", "Anesthesiology", "F", "1995-10-12", "123-456-7801"),
    ("dr.hall@example.com", "Daniel", "Hall", "Emergency Medicine", "M", "1977-09-09", "123-456-7802"),
    ("dr.allen@example.com", "Olivia", "Allen", "Psychiatry", "F", "1986-06-06", "123-456-7803"),
    ("dr.young@example.com", "Christopher", "Young", "Endocrinology", "M", "1981-03-03", "123-456-7804"),
    ("dr.king@example.com", "Isabella", "King", "Gastroenterology", "F", "1993-07-07", "123-456-7805"),
    ("dr.scott@example.com", "Matthew", "Scott", "Nephrology", "M", "1984-11-11", "123-456-7806"),
    ("dr.green@example.com", "Emma", "Green", "Rheumatology", "F", "1991-01-01", "123-456-7807"),
    ("dr.adams@example.com", "Joshua", "Adams", "Urology", "M", "1979-12-12", "123-456-7808"),
    ("dr.baker@example.com", "Sophia", "Baker", "Pulmonology", "F", "1987-08-08", "123-456-7809"),
    ("dr.carter@example.com", "Benjamin", "Carter", "Ophthalmology", "M", "1989-05-05", "123-456-7810"),
    ("dr.mitchell@example.com", "Ava", "Mitchell", "Pathology", "F", "1994-04-04", "123-456-7811")
]

async def main() -> None:
    print("Connecting to database at:", settings.database_url)
    engine = create_async_engine(str(settings.database_url))
    async_session = sessionmaker(engine, class_=AsyncSession)

    async with async_session() as db:
        for email, first_name, last_name, specialty, sex, dob, phone in DOCTORS:
            # Convert dob from string to datetime.date
            dob_date = datetime.strptime(dob, "%Y-%m-%d").date()

            # Check if doctor already exists to avoid duplicates
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
                specialty=specialty,
                sex=sex,  # Use single-character values
                dob=dob_date,  # Use the converted date object
                phone=phone
            )
            db.add(user)
            print(f"Added doctor: {first_name} {last_name} ({email}), specialty: {specialty}, sex: {sex}, dob: {dob_date}, phone: {phone}")

        await db.commit()
        print("Doctors successfully added to the database.")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
