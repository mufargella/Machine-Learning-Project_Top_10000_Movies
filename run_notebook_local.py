from __future__ import annotations

import asyncio
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


class VerboseNotebookClient(NotebookClient):
    async def execute_cell(self, cell, cell_index: int, execution_count=None, store_history: bool = True):
        if getattr(cell, "cell_type", None) == "code":
            total = len(getattr(self.nb, "cells", []) or [])
            print(f"Executing code cell {cell_index + 1}/{total}...", flush=True)
        return await super().execute_cell(
            cell,
            cell_index,
            execution_count=execution_count,
            store_history=store_history,
        )


def main() -> int:
    nb_path = Path(__file__).resolve().parent / "Ml_Project.ipynb"
    out_path = nb_path.with_name("Ml_Project.executed.ipynb")

    print(f"Starting notebook execution: {nb_path}", flush=True)

    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")

    nb = nbformat.read(nb_path, as_version=4)

    print("Notebook loaded; starting kernel execution...", flush=True)

    # zmq on Windows is most reliable with the selector event loop.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    client = VerboseNotebookClient(
        nb,
        kernel_name="python3",
        timeout=1800,
        allow_errors=False,
        resources={"metadata": {"path": str(nb_path.parent)}},
    )

    try:
        client.execute()
    except CellExecutionError:
        # Save partial output for debugging.
        nbformat.write(nb, out_path)
        print(f"Notebook execution FAILED. Partial output saved to: {out_path}")
        raise

    nbformat.write(nb, out_path)
    print(f"Notebook execution OK. Output written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
