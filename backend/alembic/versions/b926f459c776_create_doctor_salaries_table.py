"""create_doctor_salaries_table

Revision ID: b926f459c776
Revises: 954d6434947c
Create Date: 2025-05-23 15:26:35.481483

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "b926f459c776"
down_revision = "954d6434947c"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "doctor_salaries",
        sa.Column("doctor_user_id", sa.Integer(), nullable=False),
        sa.Column(
            "base_salary_annual", sa.Numeric(precision=12, scale=2), nullable=False
        ),
        sa.Column(
            "last_bonus_amount", sa.Numeric(precision=10, scale=2), nullable=True
        ),
        sa.Column("last_bonus_date", sa.Date(), nullable=True),
        sa.Column("last_bonus_reason", sa.Text(), nullable=True),
        sa.Column(
            "last_raise_percentage", sa.Numeric(precision=5, scale=2), nullable=True
        ),
        sa.Column("last_raise_date", sa.Date(), nullable=True),
        sa.Column("last_raise_reason", sa.Text(), nullable=True),
        sa.Column("next_review_period", sa.String(length=50), nullable=True),
        sa.ForeignKeyConstraint(
            ["doctor_user_id"],
            ["users.id"],
            name=op.f("fk_doctor_salaries_doctor_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("doctor_user_id", name=op.f("pk_doctor_salaries")),
    )


def downgrade():
    op.drop_table("doctor_salaries")
