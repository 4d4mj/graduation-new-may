"""add_status_and_gcal_id_to_appointments

Revision ID: 294087a693a4
Revises: b926f459c776
Create Date: 2025-05-28 19:56:22.057208

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "294087a693a4"
down_revision = "b926f459c776"
branch_labels = None
depends_on = None


def upgrade():
    # Add 'status' column
    op.add_column(
        "appointments",
        sa.Column(
            "status", sa.String(length=50), nullable=False, server_default="scheduled"
        ),
    )

    # Add 'google_calendar_event_id' column
    op.add_column(
        "appointments",
        sa.Column("google_calendar_event_id", sa.String(length=255), nullable=True),
    )

    # Create index for 'google_calendar_event_id'
    op.create_index(
        op.f("ix_appointments_google_calendar_event_id"),
        "appointments",
        ["google_calendar_event_id"],
        unique=False,
    )


def downgrade():
    pass
