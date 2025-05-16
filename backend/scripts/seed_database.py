# backend/scripts/seed_database.py
import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import random  # For generating varied data


from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, select  # For raw SQL if needed for clearing

# Add project root to sys.path to allow importing from app
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.auth import get_password_hash  # For hashing passwords
from app.config.settings import settings as app_settings  # For DB URL
from app.db.models import (
    UserModel,
    DoctorModel,
    PatientModel,
    AppointmentModel,
    AllergyModel,
)
from app.db.base import (
    Base,
)  # To create tables if they don't exist (optional, Alembic should handle it)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("seed_database")

# --- Configuration for Seed Data ---
NUM_DOCTORS = 5
NUM_PATIENTS_PER_DOCTOR = 4  # Each doctor will have roughly this many patients they see
NUM_TOTAL_PATIENTS = NUM_DOCTORS * NUM_PATIENTS_PER_DOCTOR  # Approximate total
NUM_APPOINTMENTS_PER_PATIENT = 2  # Each patient gets a couple of appointments
COMMON_PASSWORD = "TestPassword123!"
COMMON_PASSWORD_HASH = get_password_hash(COMMON_PASSWORD)

DOCTOR_SPECIALTIES = [
    "Cardiology",
    "Neurology",
    "Pediatrics",
    "Orthopedics",
    "Dermatology",
    "Oncology",
    "Radiology",
    "Psychiatry",
    "General Surgery",
    "Family Medicine",
]
PATIENT_FIRST_NAMES = [
    "Liam",
    "Olivia",
    "Noah",
    "Emma",
    "Oliver",
    "Ava",
    "Elijah",
    "Charlotte",
    "William",
    "Sophia",
    "James",
    "Amelia",
    "Benjamin",
    "Isabella",
    "Lucas",
    "Mia",
]
PATIENT_LAST_NAMES = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Rodriguez",
    "Martinez",
    "Hernandez",
    "Lopez",
    "Gonzalez",
    "Wilson",
]
ALLERGY_SUBSTANCES = [
    "Penicillin",
    "Sulfa Drugs",
    "Aspirin",
    "Ibuprofen",
    "Codeine",
    "Peanuts",
    "Tree Nuts",
    "Milk",
    "Eggs",
    "Soy",
    "Wheat",
    "Fish",
    "Shellfish",
    "Latex",
    "Bee Stings",
]
ALLERGY_REACTIONS = [
    "Hives",
    "Rash",
    "Anaphylaxis",
    "Difficulty Breathing",
    "Swelling",
    "Nausea",
    "Itching",
    "Stomach Pain",
]
ALLERGY_SEVERITIES = ["Mild", "Moderate", "Severe"]


# --- Helper Functions ---
def random_dob(start_year=1950, end_year=2005) -> date:
    year = random.randint(start_year, end_year)
    month = random.randint(1, 12)
    day = random.randint(1, 28)  # Keep it simple, avoid month-specific day counts
    return date(year, month, day)


def random_phone() -> str:
    return f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"


def random_address(i: int) -> str:
    return f"{random.randint(100, 9999)} Main St, Apt {i}, Anytown, USA"


async def clear_data(db: AsyncSession):
    logger.warning("Clearing existing data from tables...")
    # Clear in reverse order of creation / dependency
    await db.execute(text("DELETE FROM allergies;"))
    await db.execute(text("DELETE FROM appointments;"))
    # Patients and Doctors are linked to Users. If users are deleted,
    # ON DELETE CASCADE should handle patients and doctors.
    # However, appointments reference users directly for patient_id and doctor_id.
    # So, it's safer to delete appointments first, then users (which cascades).
    await db.execute(
        text("DELETE FROM patients;")
    )  # Will cascade from users if users deleted first
    await db.execute(
        text("DELETE FROM doctors;")
    )  # Will cascade from users if users deleted first
    await db.execute(text("DELETE FROM users;"))
    await db.commit()
    logger.info("Data cleared.")


async def seed_all_data(db: AsyncSession):
    created_doctors_user_ids: List[int] = []
    created_patients_user_ids: List[int] = []

    # 1. Seed Doctors (and their User entries)
    logger.info(f"Seeding {NUM_DOCTORS} doctors...")
    for i in range(NUM_DOCTORS):
        first_name = f"DoctorFirst{i + 1}"
        last_name = f"DoctorLast{i + 1}"
        email = f"doctor{i + 1}@example.com"
        specialty = random.choice(DOCTOR_SPECIALTIES)
        sex = random.choice(["M", "F"])
        dob = random_dob(1960, 1985)
        phone = random_phone()

        # Check if user already exists
        existing_user_res = await db.execute(
            select(UserModel).where(UserModel.email == email)
        )
        if existing_user_res.scalar_one_or_none():
            logger.info(f"Doctor user {email} already exists, skipping.")
            # If you want to fetch the ID to use later, you'd query here
            # For simplicity, this seed script assumes fresh creation or skips.
            continue

        user = UserModel(email=email, password_hash=COMMON_PASSWORD_HASH, role="doctor")
        db.add(user)
        await db.flush()  # Get the user.id

        doctor_profile = DoctorModel(
            user_id=user.id,
            first_name=first_name,
            last_name=last_name,
            specialty=specialty,
            sex=sex,
            dob=dob,
            phone=phone,
        )
        db.add(doctor_profile)
        created_doctors_user_ids.append(user.id)
        logger.info(
            f"  Created doctor: {first_name} {last_name} ({email}), User ID: {user.id}"
        )
    await db.commit()

    # 2. Seed Patients (and their User entries)
    logger.info(f"Seeding {NUM_TOTAL_PATIENTS} patients...")
    for i in range(NUM_TOTAL_PATIENTS):
        first_name = random.choice(PATIENT_FIRST_NAMES)
        last_name = random.choice(PATIENT_LAST_NAMES)
        email = f"patient{i + 1}@example.com"
        sex = random.choice(["M", "F"])
        dob = random_dob(1950, 2015)
        phone = random_phone()
        address = random_address(i + 1)

        existing_user_res = await db.execute(
            select(UserModel).where(UserModel.email == email)
        )
        if existing_user_res.scalar_one_or_none():
            logger.info(f"Patient user {email} already exists, skipping.")
            continue

        user = UserModel(
            email=email, password_hash=COMMON_PASSWORD_HASH, role="patient"
        )
        db.add(user)
        await db.flush()  # Get the user.id

        patient_profile = PatientModel(
            user_id=user.id,
            first_name=first_name,
            last_name=last_name,
            sex=sex,
            dob=dob,
            phone=phone,
            address=address,
        )
        db.add(patient_profile)
        created_patients_user_ids.append(user.id)
        logger.info(
            f"  Created patient: {first_name} {last_name} ({email}), User ID: {user.id}"
        )
    await db.commit()

    if not created_doctors_user_ids or not created_patients_user_ids:
        logger.warning(
            "No doctors or patients were created/found, cannot seed appointments or allergies."
        )
        return

    # 3. Seed Appointments (linking doctors and patients)
    logger.info(f"Seeding appointments...")
    appointment_count = 0
    for patient_user_id in created_patients_user_ids:
        for _ in range(NUM_APPOINTMENTS_PER_PATIENT):
            # Assign patient to a random doctor from the created list
            doctor_user_id = random.choice(created_doctors_user_ids)

            # Generate random appointment time in the near future or past
            days_offset = random.randint(
                -30, 60
            )  # Appointments from last month to next two months
            hour = random.randint(8, 16)  # 8 AM to 4 PM
            minute = random.choice([0, 30])

            # Ensure starts_at is timezone-aware (UTC)
            starts_at = datetime.now(timezone.utc).replace(
                hour=hour, minute=minute, second=0, microsecond=0
            ) + timedelta(days=days_offset)
            ends_at = starts_at + timedelta(minutes=30)
            location = f"Clinic {random.choice(['A', 'B', 'C'])}"
            notes = f"Reason: {random.choice(['Check-up', 'Follow-up', 'Consultation', 'Headache', 'Flu symptoms', 'Routine Visit'])}"

            appointment = AppointmentModel(
                patient_id=patient_user_id,
                doctor_id=doctor_user_id,
                starts_at=starts_at,
                ends_at=ends_at,
                location=location,
                notes=notes,
            )
            db.add(appointment)
            appointment_count += 1
            # logger.info(f"  Scheduled appointment for patient {patient_user_id} with doctor {doctor_user_id} at {starts_at}")
    await db.commit()
    logger.info(f"Seeded {appointment_count} appointments.")

    # 4. Seed Allergies for some patients
    logger.info(f"Seeding allergies...")
    allergy_count = 0
    # Seed allergies for a subset of patients
    for patient_user_id in random.sample(
        created_patients_user_ids,
        k=min(len(created_patients_user_ids), NUM_TOTAL_PATIENTS // 2),
    ):  # Allergies for half the patients
        num_allergies_for_patient = random.randint(
            0, 3
        )  # 0 to 3 allergies per selected patient
        if (
            num_allergies_for_patient == 0 and allergy_count == 0
        ):  # Ensure at least one patient has an allergy for testing
            num_allergies_for_patient = 1

        for _ in range(num_allergies_for_patient):
            allergy = AllergyModel(
                patient_id=patient_user_id,
                substance=random.choice(ALLERGY_SUBSTANCES),
                reaction=random.choice(ALLERGY_REACTIONS),
                severity=random.choice(ALLERGY_SEVERITIES),
            )
            db.add(allergy)
            allergy_count += 1
            # logger.info(f"  Added allergy '{allergy.substance}' for patient {patient_user_id}")
    await db.commit()
    logger.info(f"Seeded {allergy_count} allergies.")

    logger.info("Database seeding completed.")


async def main(should_clear: bool):
    logger.info(f"Connecting to database at: {app_settings.database_url}")
    engine = create_async_engine(str(app_settings.database_url))

    # Optional: Create tables if they don't exist.
    # Alembic should handle this, but this is a fallback for direct script execution.
    # async with engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.create_all)

    AsyncSessionLocal = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with AsyncSessionLocal() as db:
        if should_clear:
            await clear_data(db)
        await seed_all_data(db)

    await engine.dispose()
    logger.info("Database connection closed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed the database with initial data.")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data from relevant tables before seeding.",
    )
    args = parser.parse_args()

    asyncio.run(main(should_clear=args.clear))
