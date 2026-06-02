import asyncio
from pathlib import Path

from config import settings
from db.connection import close_pool, get_conn, init_pool
from repositories.user_repository import UserRepository


def _write_user_id_to_env(user_id: str) -> None:
    env_path = Path(".env")
    if not env_path.exists():
        sample = Path(".env.example")
        if sample.exists():
            env_path.write_text(sample.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            env_path.write_text("", encoding="utf-8")

    lines = env_path.read_text(encoding="utf-8").splitlines()
    updated = False
    new_lines = []
    for line in lines:
        if line.startswith("USER_ID="):
            new_lines.append(f"USER_ID={user_id}")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f"USER_ID={user_id}")
    env_path.write_text("\n".join(new_lines).strip() + "\n", encoding="utf-8")


async def main():
    await init_pool(settings.database_url)
    async with get_conn() as conn:
        repo = UserRepository(conn)
        user = await repo.create()
    await close_pool()
    _write_user_id_to_env(str(user["id"]))
    print(f"Created user {user['id']} for {settings.user_name}.")
    print("USER_ID has been written to .env.")


if __name__ == "__main__":
    asyncio.run(main())
