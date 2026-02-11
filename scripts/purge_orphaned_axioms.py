import asyncio

from graph_rlm.backend.src.mcp_integration.skill_storage import get_axioms_manager


async def purge_orphaned_axioms():
    mgr = get_axioms_manager()

    # 1. Get all axioms from DB
    query = "MATCH (a:Axiom) RETURN a.name as name"
    results = mgr.db.query(query)
    db_axioms = {r["name"] for r in results} if results else set()

    print(f"Found {len(db_axioms)} axioms in DB: {db_axioms}")

    # 2. Get all axioms on disk (active)
    active_axioms = set()
    for item in mgr.axioms_dir.iterdir():
        if item.is_file() and item.suffix == ".py" and not item.name.startswith("__"):
            active_axioms.add(item.stem)

    print(f"Found {len(active_axioms)} active axioms on disk: {active_axioms}")

    # 3. Identify orphans
    orphans = db_axioms - active_axioms

    # 4. Delete orphans
    if orphans:
        print(f"Purging {len(orphans)} orphaned axioms from DB: {orphans}")
        for name in orphans:
            mgr.db.query("MATCH (a:Axiom {name: $name}) DELETE a", {"name": name})
            print(f"Deleted {name}")
    else:
        print("No orphaned axioms found.")


if __name__ == "__main__":
    asyncio.run(purge_orphaned_axioms())
