# backend/scripts/seed_database.py
import asyncio
import logging
from datetime import date, datetime, timedelta, timezone as TZ
from typing import List, Dict, Any, Optional
import random  # For generating varied data


from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, select  # For raw SQL if needed for clearing
from sqlalchemy.exc import IntegrityError

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
NUM_DOCTORS = 30
NUM_PATIENTS_PER_DOCTOR = 5  # Each doctor will have roughly this many patients they see
NUM_TOTAL_PATIENTS = NUM_DOCTORS * NUM_PATIENTS_PER_DOCTOR  # Approximate total
NUM_APPOINTMENTS_PER_PATIENT = 3  # Each patient gets a couple of appointments
COMMON_PASSWORD = "TestPassword123!"
COMMON_PASSWORD_HASH = get_password_hash(COMMON_PASSWORD)

DOCTOR_FIRST_NAMES = [
    "Ahmad",
    "Mohamad",
    "Ali",
    "Hassan",
    "Hussein",
    "Omar",
    "Karim",
    "Jad",
    "Rami",
    "Samir",
    "Walid",
    "Fadi",
    "Nadim",
    "Ziad",
    "Tarek",
    "Elie",
    "Georges",
    "Tony",
    "Charbel",
    "Rami",
    "Bassam",
    "Nabil",
    "Fouad",
    "Sami",
    "Marwan",
    "Rafic",
    "Michel",
    "Antoine",
    "Joseph",
    "Sleiman",
    "Rita",
    "Maya",
    "Layal",
    "Nour",
    "Sara",
    "Lina",
    "Mira",
    "Yara",
    "Dina",
    "Rana",
    "Hiba",
    "Racha",
    "Mariam",
    "Joumana",
    "Nadine",
    "Rola",
    "Samar",
    "Hind",
    "Zeina",
    "Carla",
    "Jana",
    "Lea",
    "Christelle",
    "Nancy",
    "Joelle",
    "Micheline",
    "Paula",
    "Sandy",
    "Ghina",
    "Farah",
]


DOCTOR_LAST_NAMES = [
    "Khoury",
    "Haddad",
    "Nassar",
    "Saad",
    "Sleiman",
    "Fares",
    "Mansour",
    "Habib",
    "Antoun",
    "Gerges",
    "Saliba",
    "Nasr",
    "Abi Nader",
    "El Hajj",
    "Bou Saab",
    "Barakat",
    "Chahine",
    "Maalouf",
    "Karam",
    "Sarkis",
    "Azar",
    "Mikhael",
    "Tannous",
    "Younes",
    "Ghanem",
    "Farhat",
    "Zein",
    "Awad",
    "Saba",
    "Rizk",
    "Hanna",
    "Jabbour",
    "Salem",
    "Fakhoury",
    "Matar",
    "Assaf",
    "Bazzi",
    "Elia",
    "Doumit",
    "Sfeir",
    "Kfoury",
    "Daou",
    "Chamoun",
    "Khalil",
    "Issa",
    "Najjar",
    "Abou Jaoude",
    "Moussa",
    "Abou Rjeily",
    "Tabet",
    "Bou Khalil",
    "Mouawad",
    "Aoun",
    "Bou Ghannam",
    "Moughnieh",
]


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
    "Gastroenterology",
    "Nephrology",
    "Hematology",
    "Rheumatology",
    "Pulmonology",
    "Ophthalmology",
    "Otolaryngology",
    "Urology",
    "Anesthesiology",
    "Emergency Medicine",
    "Infectious Disease",
    "Geriatrics",
    "Obstetrics and Gynecology",
    "Plastic Surgery",
    "Pathology",
    "Immunology",
    "Sports Medicine",
    "Pain Medicine",
    "Sleep Medicine",
    "Rehabilitation Medicine",
    "Vascular Surgery",
]

PATIENT_FIRST_NAMES = [
    "Ahmad",
    "Mohamad",
    "Ali",
    "Hassan",
    "Hussein",
    "Omar",
    "Karim",
    "Jad",
    "Rami",
    "Samir",
    "Walid",
    "Fadi",
    "Nadim",
    "Ziad",
    "Tarek",
    "Rami",
    "Elie",
    "Georges",
    "Tony",
    "Charbel",
    "Rita",
    "Maya",
    "Layal",
    "Nour",
    "Sara",
    "Lina",
    "Mira",
    "Yara",
    "Dina",
    "Rana",
    "Hiba",
    "Racha",
    "Mariam",
    "Joumana",
    "Nadine",
    "Rola",
    "Samar",
    "Hind",
    "Zeina",
    "Carla",
    "Jana",
    "Lea",
    "Christelle",
    "Nancy",
    "Joelle",
    "Micheline",
    "Paula",
    "Sandy",
    "Ghina",
    "Farah",
]

PATIENT_LAST_NAMES = [
    "Khoury",
    "Haddad",
    "Nassar",
    "Saad",
    "Sleiman",
    "Fares",
    "Mansour",
    "Habib",
    "Antoun",
    "Gerges",
    "Saliba",
    "Nasr",
    "Abi Nader",
    "El Hajj",
    "Bou Saab",
    "Barakat",
    "Chahine",
    "Maalouf",
    "Karam",
    "Sarkis",
    "Azar",
    "Mikhael",
    "Tannous",
    "Younes",
    "Ghanem",
    "Farhat",
    "Zein",
    "Awad",
    "Saba",
    "Rizk",
    "Hanna",
    "Jabbour",
    "Salem",
    "Fakhoury",
    "Matar",
    "Assaf",
    "Bazzi",
    "Elia",
    "Doumit",
    "Sfeir",
    "Kfoury",
    "Saba",
    "Daou",
    "Chamoun",
    "Khalil",
    "Issa",
    "Najjar",
    "Abou Jaoude",
    "Moussa",
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
    "Cats",
    "Dogs",
    "Mold",
    "Grass",
    "Insect Stings",
    "Sesame",
    "Strawberries",
    "Tomatoes",
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
    "Coughing",
    "Runny Nose",
    "Watery Eyes",
    "Vomiting",
    "Diarrhea",
    "Dizziness",
    "Chest Tightness",
    "Wheezing",
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
    "Blood Pressure Check",
    "Diabetes Management",
    "Prescription Refill",
    "Stomach Pain",
    "Back Pain",
    "Cough",
    "Shortness of Breath",
    "Dizziness",
    "Annual Physical",
    "Lab Results Review",
    "Chest Pain",
    "Allergy Symptoms",
    "Ear Infection",
    "Sore Throat",
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
        first_name = random.choice(DOCTOR_FIRST_NAMES)
        last_name = random.choice(DOCTOR_LAST_NAMES)
        email = f"doctor.{first_name.lower()}.{last_name.lower()}{i + 1}@example.com".replace(
            " ", ""
        )

        existing_user_res = await db.execute(
            select(UserModel.id).where(UserModel.email == email)
        )
        existing_user_id = existing_user_res.scalar_one_or_none()

        if existing_user_id:
            logger.info(
                f"Doctor user {email} already exists with ID {existing_user_id}, using existing."
            )
            if (
                existing_user_id not in created_doctors_user_ids
            ):  # Ensure no duplicates in our list
                created_doctors_user_ids.append(existing_user_id)
            continue

        user = UserModel(email=email, password_hash=COMMON_PASSWORD_HASH, role="doctor")
        db.add(user)
        try:
            await (
                db.flush()
            )  # Get user.id before commit to avoid issues if commit fails later
            doctor_profile = DoctorModel(
                user_id=user.id,
                first_name=first_name,
                last_name=last_name,
                specialty=random.choice(DOCTOR_SPECIALTIES),
                sex=random.choice(["M", "F"]),
                dob=random_dob(1960, 1990),
                phone=random_phone(),
            )
            db.add(doctor_profile)
            created_doctors_user_ids.append(user.id)
            logger.info(
                f"  Added to session: Dr. {first_name} {last_name} ({email}), User ID: {user.id}"
            )
        except Exception as e:  # Catch potential errors during add/flush
            await db.rollback()
            logger.error(f"Error adding/flushing doctor {email}: {e}", exc_info=True)
            continue  # Skip this doctor

    try:
        await db.commit()  # Commit all doctors once
        logger.info(f"Committed {len(created_doctors_user_ids)} doctors.")
    except IntegrityError as e:
        await db.rollback()
        logger.error(f"Integrity error committing doctors: {e.orig}", exc_info=True)
    except Exception as e:
        await db.rollback()
        logger.error(f"Unexpected error committing doctors: {e}", exc_info=True)

    # 2. Seed Patients
    logger.info(f"Seeding {NUM_TOTAL_PATIENTS} patients...")
    for i in range(NUM_TOTAL_PATIENTS):
        first_name = random.choice(PATIENT_FIRST_NAMES)
        if i == 0:
            first_name = "Alice"
            last_name = "Wonder"
        elif i == 1:
            first_name = "Alice"
            last_name = "Smith"
        else:
            last_name = random.choice(PATIENT_LAST_NAMES)

        email = f"patient.{first_name.lower()}.{last_name.lower()}{i + 1}@example.com".replace(
            " ", ""
        )

        existing_user_res = await db.execute(
            select(UserModel.id).where(UserModel.email == email)
        )
        existing_user_id = existing_user_res.scalar_one_or_none()

        if existing_user_id:
            logger.info(
                f"Patient user {email} already exists with ID {existing_user_id}, using existing."
            )
            if existing_user_id not in created_patients_user_ids:
                created_patients_user_ids.append(existing_user_id)
            patient_id_to_name_map[existing_user_id] = f"{first_name} {last_name}"
            continue

        user = UserModel(
            email=email, password_hash=COMMON_PASSWORD_HASH, role="patient"
        )
        db.add(user)
        try:
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
                f"  Added to session: Patient {first_name} {last_name} ({email}), User ID: {user.id}"
            )
        except Exception as e:
            await db.rollback()
            logger.error(f"Error adding/flushing patient {email}: {e}", exc_info=True)
            continue

    try:
        await db.commit()  # Commit all patients once
        logger.info(f"Committed {len(created_patients_user_ids)} patients.")
    except IntegrityError as e:
        await db.rollback()
        logger.error(f"Integrity error committing patients: {e.orig}", exc_info=True)
    except Exception as e:
        await db.rollback()
        logger.error(f"Unexpected error committing patients: {e}", exc_info=True)

    if not created_doctors_user_ids or not created_patients_user_ids:
        logger.error(
            "Critical: No doctors or patients available for seeding appointments/allergies. Exiting dependent seeding."
        )
        return

    # 3. Seed Appointments
    logger.info(f"Seeding appointments with individual commits and error handling...")
    attempted_appointment_count = 0
    successful_appointment_inserts = 0
    failed_appointment_inserts = 0
    now_utc_dt = datetime.now(TZ.utc)

    doctor_for_pagination_test = (
        created_doctors_user_ids[0] if created_doctors_user_ids else None
    )

    for i, patient_user_id in enumerate(created_patients_user_ids):
        if not doctor_for_pagination_test:  # Safety check
            assigned_doctor_id = (
                random.choice(created_doctors_user_ids)
                if created_doctors_user_ids
                else None
            )
        elif i < (
            NUM_TOTAL_PATIENTS * 0.7
        ):  # Ensure enough appointments for the test doctor
            assigned_doctor_id = doctor_for_pagination_test
        else:
            assigned_doctor_id = (
                random.choice(created_doctors_user_ids)
                if created_doctors_user_ids
                else None
            )

        if not assigned_doctor_id:  # If no doctors, skip
            continue

        alice_wonder_id = next(
            (
                pid
                for pid, name in patient_id_to_name_map.items()
                if name == "Alice Wonder"
            ),
            None,
        )
        alice_smith_id = next(
            (
                pid
                for pid, name in patient_id_to_name_map.items()
                if name == "Alice Smith"
            ),
            None,
        )

        date_scenarios = []
        if patient_user_id == alice_wonder_id:
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
                ("today", now_utc_dt, "Today's check for Alice Wonder"),
                (
                    "specific_future_1",
                    datetime(2025, 6, 22, tzinfo=TZ.utc),
                    "Specific future date 1 for Alice",
                ),
                (
                    "specific_future_2",
                    datetime(2025, 6, 22, tzinfo=TZ.utc),
                    "Specific future date 2 for Alice - different time",
                ),
            ]
        elif patient_user_id == alice_smith_id:
            date_scenarios = [
                (
                    "upcoming_4_days",
                    now_utc_dt + timedelta(days=4),
                    "Check-up for Alice Smith",
                ),
            ]
        else:
            num_appts = random.randint(1, NUM_APPOINTMENTS_PER_PATIENT)
            for _ in range(num_appts):
                days_offset = random.randint(-60, 60)
                date_scenarios.append(
                    (
                        f"random_offset_{days_offset}",
                        now_utc_dt + timedelta(days=days_offset),
                        random.choice(APPOINTMENT_NOTES_KEYWORDS),
                    )
                )

        # To reduce collisions, keep track of (doctor_id, starts_at) for the current batch being processed
        # This is a local check for the current seeder run, not a DB check.
        local_doctor_day_slots = {}  # doctor_id -> {date_str -> set_of_start_hours_minutes}

        for desc, appt_datetime_base_utc, note_reason in date_scenarios:
            attempted_appointment_count += 1

            # Try to find a unique slot for this doctor on this day locally
            slot_found_locally = False
            for attempt in range(
                10
            ):  # Try up to 10 times to find a unique slot for this appt
                hour = random.randint(8, 16)
                minute_options = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
                minute = random.choice(minute_options)

                current_starts_at = appt_datetime_base_utc.replace(
                    hour=hour, minute=minute, second=0, microsecond=0
                )

                date_key = current_starts_at.date().isoformat()
                time_key = (current_starts_at.hour, current_starts_at.minute)

                if assigned_doctor_id not in local_doctor_day_slots:
                    local_doctor_day_slots[assigned_doctor_id] = {}
                if date_key not in local_doctor_day_slots[assigned_doctor_id]:
                    local_doctor_day_slots[assigned_doctor_id][date_key] = set()

                if time_key not in local_doctor_day_slots[assigned_doctor_id][date_key]:
                    local_doctor_day_slots[assigned_doctor_id][date_key].add(time_key)
                    slot_found_locally = True
                    break  # Found a locally unique slot for this doctor/day/time
                # If not found, loop will try a new random time

            if not slot_found_locally:
                logger.warning(
                    f"Could not find a locally unique slot for doctor {assigned_doctor_id} on {date_key} after 10 attempts. Skipping this specific appointment."
                )
                failed_appointment_inserts += 1
                continue  # Skip this appointment if no locally unique slot found

            starts_at = current_starts_at  # Use the slot found
            duration_options_minutes = [15, 30, 45, 60]
            ends_at = starts_at + timedelta(
                minutes=random.choice(duration_options_minutes)
            )

            appointment = AppointmentModel(
                patient_id=patient_user_id,
                doctor_id=assigned_doctor_id,
                starts_at=starts_at,
                ends_at=ends_at,
                location=f"Clinic {random.choice(['A', 'B', 'C'])}",
                notes=note_reason,
            )
            db.add(appointment)
            try:
                await db.commit()
                successful_appointment_inserts += 1
            except IntegrityError as e:
                await db.rollback()
                logger.warning(
                    f"Skipped duplicate appointment for doctor {assigned_doctor_id} at {starts_at} due to DB unique constraint. Error: {e.orig.diag.message_detail if hasattr(e.orig, 'diag') else e.orig}"
                )
                failed_appointment_inserts += 1
            except Exception as e_other:
                await db.rollback()
                logger.error(
                    f"Error committing appointment for doctor {assigned_doctor_id} at {starts_at}: {e_other}",
                    exc_info=True,
                )
                failed_appointment_inserts += 1

    logger.info(f"Attempted to seed {attempted_appointment_count} appointments.")
    logger.info(f"Successfully seeded {successful_appointment_inserts} appointments.")
    if failed_appointment_inserts > 0:
        logger.warning(
            f"Failed/Skipped {failed_appointment_inserts} appointments due to conflicts or errors."
        )

    # 4. Seed Allergies
    logger.info(f"Seeding allergies for a subset of patients...")
    allergy_count = 0
    # ... (rest of your allergy seeding logic, which seems fine as it doesn't have unique constraints like appointments) ...
    # Make sure to commit after adding allergies
    patients_for_allergies = set()
    alice_wonder_id = next(
        (pid for pid, name in patient_id_to_name_map.items() if name == "Alice Wonder"),
        None,
    )
    alice_smith_id = next(
        (pid for pid, name in patient_id_to_name_map.items() if name == "Alice Smith"),
        None,
    )

    if alice_wonder_id:
        patients_for_allergies.add(alice_wonder_id)
    if alice_smith_id:
        patients_for_allergies.add(alice_smith_id)

    num_other_patients_with_allergies = NUM_TOTAL_PATIENTS // 3
    if len(created_patients_user_ids) > len(patients_for_allergies):
        remaining_patient_ids = list(
            set(created_patients_user_ids) - patients_for_allergies
        )
        if remaining_patient_ids:  # Check if list is not empty
            patients_to_sample = min(
                len(remaining_patient_ids), num_other_patients_with_allergies
            )
            if patients_to_sample > 0:
                patients_for_allergies.update(
                    random.sample(remaining_patient_ids, k=patients_to_sample)
                )

    for patient_user_id in patients_for_allergies:
        num_allergies_for_patient = 0
        if patient_user_id == alice_wonder_id:
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
            num_allergies_for_patient = 2
        elif patient_user_id == alice_smith_id:
            db.add(
                AllergyModel(
                    patient_id=patient_user_id,
                    substance="Dust Mites",
                    reaction="Sneezing",
                    severity="Mild",
                )
            )
            num_allergies_for_patient = 1
        else:
            num_allergies_for_patient = random.randint(1, 3)
            for _ in range(num_allergies_for_patient):
                db.add(
                    AllergyModel(
                        patient_id=patient_user_id,
                        substance=random.choice(ALLERGY_SUBSTANCES),
                        reaction=random.choice(ALLERGY_REACTIONS),
                        severity=random.choice(ALLERGY_SEVERITIES),
                    )
                )
        allergy_count += num_allergies_for_patient

    try:
        await db.commit()  # Commit all allergies
        logger.info(f"Seeded {allergy_count} allergies.")
    except Exception as e:
        await db.rollback()
        logger.error(f"Error committing allergies: {e}", exc_info=True)

    logger.info("Database seeding completed.")


async def main(should_clear: bool):
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
