# backend/scripts/seed_database.py
import asyncio
import logging
from datetime import date, datetime, timedelta, timezone as TZ
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

from app.core.auth import get_password_hash
from app.config.settings import settings as app_settings
from app.db.models import (
    UserModel,
    DoctorModel,
    PatientModel,
    AppointmentModel,
    AllergyModel,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, select


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("seed_database")

# --- Configuration for Seed Data ---
NUM_DOCTORS = 10
NUM_PATIENTS_PER_DOCTOR = 5  # Each doctor will have roughly this many patients they see
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
    "Endocrinology",
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
    "Henry",
    "Evelyn",
    "Alexander",
    "Harper",
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
    "Anderson",
    "Thomas",
    "Taylor",
    "Moore",
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
    "Pollen",
    "Dust Mites",
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
    "Headache",
    "Sneezing",
]
ALLERGY_SEVERITIES = ["Mild", "Moderate", "Severe"]

APPOINTMENT_NOTES_KEYWORDS = [
    "Headache",
    "Fever",
    "Check-up",
    "Follow-up",
    "Consultation",
    "Cardiac Concerns",
    "Joint Pain",
    "Fatigue",
    "Flu Symptoms",
    "Vaccination",
    "Skin Rash",
]


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
    await db.execute(text("DELETE FROM allergies;"))
    await db.execute(text("DELETE FROM appointments;"))
    await db.execute(text("DELETE FROM patients;"))
    await db.execute(text("DELETE FROM doctors;"))
    await db.execute(
        text("DELETE FROM users WHERE role IN ('patient', 'doctor');")
    )  # Keep admin if any
    await db.commit()
    logger.info("Relevant data cleared.")


async def seed_all_data(db: AsyncSession):
    created_doctors_user_ids: List[int] = []
    created_patients_user_ids: List[int] = []
    patient_id_to_name_map: Dict[int, str] = {}

    # 1. Seed Doctors
    logger.info(f"Seeding {NUM_DOCTORS} doctors...")
    for i in range(NUM_DOCTORS):
        first_name = f"DocFirst{i + 1}"
        last_name = f"DocLast{i + 1}"
        email = f"doctor{i + 1}@example.com"

        existing_user_res = await db.execute(
            select(UserModel.id).where(UserModel.email == email)
        )
        existing_user_id = existing_user_res.scalar_one_or_none()

        if existing_user_id:
            logger.info(
                f"Doctor user {email} already exists with ID {existing_user_id}, using existing."
            )
            created_doctors_user_ids.append(existing_user_id)
            # Optionally, fetch and update doctor profile if needed, or assume it's fine
            continue

        user = UserModel(email=email, password_hash=COMMON_PASSWORD_HASH, role="doctor")
        db.add(user)
        await db.flush()

        doctor_profile = DoctorModel(
            user_id=user.id,
            first_name=first_name,
            last_name=last_name,
            specialty=random.choice(DOCTOR_SPECIALTIES),
            sex=random.choice(["M", "F"]),
            dob=random_dob(1960, 1985),
            phone=random_phone(),
        )
        db.add(doctor_profile)
        created_doctors_user_ids.append(user.id)
        logger.info(
            f"  Created doctor: {first_name} {last_name} ({email}), User ID: {user.id}"
        )
    await db.commit()

    # 2. Seed Patients
    logger.info(f"Seeding {NUM_TOTAL_PATIENTS} patients...")
    for i in range(NUM_TOTAL_PATIENTS):
        first_name = random.choice(PATIENT_FIRST_NAMES)
        # Introduce a couple of patients with the same first name for ambiguity testing
        if i == 0:
            first_name = "Alice"
            last_name = "Wonder"
        elif i == 1:
            first_name = "Alice"
            last_name = "Smith"
        else:
            last_name = random.choice(PATIENT_LAST_NAMES)

        email = f"patient{i + 1}@example.com"

        existing_user_res = await db.execute(
            select(UserModel.id).where(UserModel.email == email)
        )
        existing_user_id = existing_user_res.scalar_one_or_none()

        if existing_user_id:
            logger.info(
                f"Patient user {email} already exists with ID {existing_user_id}, using existing."
            )
            created_patients_user_ids.append(existing_user_id)
            patient_id_to_name_map[existing_user_id] = (
                f"{first_name} {last_name}"  # Store name
            )
            # Optionally, fetch and update patient profile
            continue

        user = UserModel(
            email=email, password_hash=COMMON_PASSWORD_HASH, role="patient"
        )
        db.add(user)
        await db.flush()

        patient_profile = PatientModel(
            user_id=user.id,
            first_name=first_name,
            last_name=last_name,
            sex=random.choice(["M", "F"]),
            dob=random_dob(1950, 2015),
            phone=random_phone(),
            address=random_address(i + 1),
        )
        db.add(patient_profile)
        created_patients_user_ids.append(user.id)
        patient_id_to_name_map[user.id] = f"{first_name} {last_name}"
        logger.info(
            f"  Created patient: {first_name} {last_name} ({email}), User ID: {user.id}"
        )
    await db.commit()

    if not created_doctors_user_ids or not created_patients_user_ids:
        logger.error(
            "Critical: No doctors or patients available for seeding appointments/allergies. Exiting."
        )
        return

    # 3. Seed Appointments - More strategic distribution
    logger.info(f"Seeding appointments with more varied dates...")
    appointment_count = 0
    now_utc_dt = datetime.now(TZ.utc)  # Use TZ alias

    # Ensure at least one doctor has many patients for pagination testing
    doctor_for_pagination_test = (
        created_doctors_user_ids[0] if created_doctors_user_ids else None
    )

    for i, patient_user_id in enumerate(created_patients_user_ids):
        # Assign most patients to the doctor_for_pagination_test
        if doctor_for_pagination_test and i < (
            NUM_TOTAL_PATIENTS * 0.7
        ):  # ~70% of patients go to this doctor
            assigned_doctor_id = doctor_for_pagination_test
        else:  # Distribute remaining patients among other doctors
            assigned_doctor_id = random.choice(created_doctors_user_ids)

        # Create varied appointments for Patient ID user_id 101 (Alice Wonderland, assuming her user_id becomes 101 based on typical DB sequences)
        # We need to find Alice Wonderland's actual created ID
        alice_wonder_id = None
        for pid, name in patient_id_to_name_map.items():
            if name == "Alice Wonder":
                alice_wonder_id = pid
                break

        alice_smith_id = None
        for pid, name in patient_id_to_name_map.items():
            if name == "Alice Smith":
                alice_smith_id = pid
                break

        date_scenarios = []
        if (
            patient_user_id == alice_wonder_id and assigned_doctor_id
        ):  # Specific scenarios for "Alice Wonder"
            date_scenarios = [
                (
                    "upcoming_3_days",
                    now_utc_dt + timedelta(days=3),
                    "Check-up for Alice Wonder",
                ),
                (
                    "past_5_days",
                    now_utc_dt - timedelta(days=5),
                    "Follow-up for Alice Wonder",
                ),
                (
                    "past_15_days",
                    now_utc_dt - timedelta(days=15),
                    "Consultation for Alice Wonder",
                ),
                (
                    "past_40_days",
                    now_utc_dt - timedelta(days=40),
                    "Old record for Alice Wonder",
                ),
                (
                    "today",
                    now_utc_dt,
                    "Today's check for Alice Wonder",
                ),  # For testing "today"
            ]
        elif (
            patient_user_id == alice_smith_id and assigned_doctor_id
        ):  # Specific for "Alice Smith" for ambiguity
            date_scenarios = [
                (
                    "upcoming_4_days",
                    now_utc_dt + timedelta(days=4),
                    "Check-up for Alice Smith",
                ),
            ]
        else:  # Generic appointments for other patients
            num_appts = random.randint(1, 3)
            for _ in range(num_appts):
                days_offset = random.randint(-60, 60)  # Wider range
                date_scenarios.append(
                    (
                        f"random_offset_{days_offset}",
                        now_utc_dt + timedelta(days=days_offset),
                        random.choice(APPOINTMENT_NOTES_KEYWORDS),
                    )
                )

        for desc, appt_datetime_utc, note_reason in date_scenarios:
            hour = random.randint(8, 16)
            minute = random.choice([0, 15, 30, 45])
            starts_at = appt_datetime_utc.replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
            ends_at = starts_at + timedelta(
                minutes=random.choice([15, 30, 45])
            )  # Varied duration

            appointment = AppointmentModel(
                patient_id=patient_user_id,
                doctor_id=assigned_doctor_id,
                starts_at=starts_at,
                ends_at=ends_at,
                location=f"Clinic {random.choice(['X', 'Y', 'Z'])}",
                notes=note_reason,
            )
            db.add(appointment)
            appointment_count += 1
    await db.commit()
    logger.info(f"Seeded {appointment_count} appointments.")

    # 4. Seed Allergies
    logger.info(f"Seeding allergies for a subset of patients...")
    allergy_count = 0
    # Ensure Alice Wonder and Alice Smith have some allergies for testing
    patients_for_allergies = set()
    if alice_wonder_id:
        patients_for_allergies.add(alice_wonder_id)
    if alice_smith_id:
        patients_for_allergies.add(alice_smith_id)

    # Add some other random patients to have allergies
    num_other_patients_with_allergies = NUM_TOTAL_PATIENTS // 3
    if len(created_patients_user_ids) > len(patients_for_allergies):
        remaining_patient_ids = list(
            set(created_patients_user_ids) - patients_for_allergies
        )
        if remaining_patient_ids:  # Check if list is not empty
            patients_for_allergies.update(
                random.sample(
                    remaining_patient_ids,
                    k=min(
                        len(remaining_patient_ids), num_other_patients_with_allergies
                    ),
                )
            )

    for patient_user_id in patients_for_allergies:
        num_allergies = 0
        if patient_user_id == alice_wonder_id:
            num_allergies = 2  # Ensure Alice Wonder has specific allergies
            db.add(
                AllergyModel(
                    patient_id=patient_user_id,
                    substance="Penicillin",
                    reaction="Hives",
                    severity="Moderate",
                )
            )
            db.add(
                AllergyModel(
                    patient_id=patient_user_id,
                    substance="Peanuts",
                    reaction="Anaphylaxis",
                    severity="Severe",
                )
            )
            allergy_count += 2
        elif patient_user_id == alice_smith_id:
            num_allergies = 1
            db.add(
                AllergyModel(
                    patient_id=patient_user_id,
                    substance="Dust Mites",
                    reaction="Sneezing",
                    severity="Mild",
                )
            )
            allergy_count += 1
        else:
            num_allergies = random.randint(
                1, 3
            )  # Other patients get 1-3 random allergies
            for _ in range(num_allergies):
                db.add(
                    AllergyModel(
                        patient_id=patient_user_id,
                        substance=random.choice(ALLERGY_SUBSTANCES),
                        reaction=random.choice(ALLERGY_REACTIONS),
                        severity=random.choice(ALLERGY_SEVERITIES),
                    )
                )
                allergy_count += 1
    await db.commit()
    logger.info(f"Seeded {allergy_count} allergies.")
    logger.info("Database seeding completed.")


async def main(should_clear: bool):
    # ... (main function remains mostly the same, ensure it calls the updated seed_all_data)
    logger.info(f"Connecting to database at: {app_settings.database_url}")
    engine = create_async_engine(str(app_settings.database_url))
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

    parser = argparse.ArgumentParser(
        description="Seed the database with enhanced initial data."
    )
    parser.add_argument(
        "--clear", action="store_true", help="Clear existing data before seeding."
    )
    args = parser.parse_args()
    asyncio.run(main(should_clear=args.clear))
