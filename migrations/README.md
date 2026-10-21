# Database Migrations

This directory contains Alembic database migration scripts for persistent verification audit logging and report generation.

### Usage:
- Generate a new migration: `alembic revision --autogenerate -m "description"`
- Apply migrations: `alembic upgrade head`
- Rollback: `alembic downgrade -1`

