"""fix user_id type from uuid to integer

Revision ID: 002
Revises: 001
Create Date: 2024-01-15 20:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade():
    # Drop and recreate the table with correct user_id type
    op.drop_table('match_jobs')
    
    op.create_table('match_jobs',
    sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('status', sa.String(length=50), nullable=False),
    sa.Column('task_id', sa.String(length=255), nullable=True),
    sa.Column('resume_filename', sa.String(length=255), nullable=False),
    sa.Column('job_ids', sa.JSON(), nullable=False),
    sa.Column('preferences', sa.JSON(), nullable=False),
    sa.Column('match_results', sa.JSON(), nullable=True),
    sa.Column('parsed_resume_id', sa.Integer(), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('progress_percentage', sa.Float(), nullable=True),
    sa.Column('current_step', sa.String(length=100), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('started_at', sa.DateTime(), nullable=True),
    sa.Column('completed_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_match_jobs_created_at'), 'match_jobs', ['created_at'], unique=False)
    op.create_index(op.f('ix_match_jobs_status'), 'match_jobs', ['status'], unique=False)
    op.create_index(op.f('ix_match_jobs_task_id'), 'match_jobs', ['task_id'], unique=False)
    op.create_index(op.f('ix_match_jobs_user_id'), 'match_jobs', ['user_id'], unique=False)


def downgrade():
    # Drop the table
    op.drop_index(op.f('ix_match_jobs_user_id'), table_name='match_jobs')
    op.drop_index(op.f('ix_match_jobs_task_id'), table_name='match_jobs')
    op.drop_index(op.f('ix_match_jobs_status'), table_name='match_jobs')
    op.drop_index(op.f('ix_match_jobs_created_at'), table_name='match_jobs')
    op.drop_table('match_jobs')