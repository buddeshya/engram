from alembic import command
from alembic.config import Config


def main() -> None:
    cfg = Config("alembic.ini")
    command.upgrade(cfg, "head")
    print("Database initialized via Alembic.")


if __name__ == "__main__":
    main()
