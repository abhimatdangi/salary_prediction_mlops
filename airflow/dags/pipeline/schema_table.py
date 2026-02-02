import sqlalchemy as sa
from sqlalchemy import create_engine

def create_bigtable(
    db_url: str = None,
    table_name: str = "raw_salary"
):

    db_url = db_url or "mysql+pymysql://app:app@mariadb:3306/analytics"
    engine = create_engine(db_url)
    metadata = sa.MetaData()

    raw = sa.Table(
        table_name, metadata,
        sa.Column('FIRST_NAME', sa.String(80)),
        sa.Column('LAST_NAME', sa.String(80)),
        sa.Column('SEX', sa.String(10)),
        sa.Column('DOJ', sa.String(20)),           # keep as string; we’ll parse later
        sa.Column('CURRENT_DATE', sa.String(20)),  # keep as string; we’ll parse later
        sa.Column('DESIGNATION', sa.String(80)),
        sa.Column('AGE', sa.Float),
        sa.Column('SALARY', sa.Float),
        sa.Column('UNIT', sa.String(80)),
        sa.Column('LEAVES_USED', sa.Float),
        sa.Column('LEAVES_REMAINING', sa.Float),
        sa.Column('RATINGS', sa.Float),
        sa.Column('PAST_EXP', sa.Float),
    )
    metadata.create_all(engine)

    with engine.connect() as conn:
        result = conn.execute(sa.text(f"SHOW TABLES LIKE '{table_name}'")).fetchall()
        print(f"Table '{table_name}' ready." if result else f"Failed to create '{table_name}'.")
