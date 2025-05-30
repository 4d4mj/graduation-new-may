"""add_is_discharged_to_appointments

Revision ID: acfd667d9ab9
Revises: 294087a693a4
Create Date: 2025-05-30 09:37:55.924371

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "acfd667d9ab9"
down_revision = "294087a693a4"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "appointments",
        sa.Column(
            "is_discharged",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )


def downgrade():
    op.drop_column("appointments", "is_discharged")
