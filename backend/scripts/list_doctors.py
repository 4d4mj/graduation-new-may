import asyncio
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow importing from app
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from app.db.models.user import UserModel
from app.db.models.doctor import DoctorModel
from app.config.settings import settings

async def main() -> None:
    print("Connecting to database at:", settings.database_url)
    engine = create_async_engine(str(settings.database_url))
    async_session = sessionmaker(engine, class_=AsyncSession)

    async with async_session() as db:
        # Query doctors using the SQLAlchemy ORM
        result = await db.execute(
            select(UserModel, DoctorModel)
            .join(DoctorModel, UserModel.id == DoctorModel.user_id)
            .where(UserModel.role == "doctor")
        )

        doctors = result.all()

        if not doctors:
            print("No doctors found in the database.")
        else:
            print(f"Found {len(doctors)} doctors in the database:")
            print("-" * 80)
            print(f"{'ID':<5} {'Name':<25} {'Email':<30} {'Specialty':<25}")
            print("-" * 80)

            for user, doctor in doctors:
                full_name = f"{doctor.first_name} {doctor.last_name}"
                print(f"{user.id:<5} {full_name:<25} {user.email:<30} {doctor.specialty:<25}")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
