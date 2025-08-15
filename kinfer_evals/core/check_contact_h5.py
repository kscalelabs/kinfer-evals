#!/usr/bin/env python3
"""Inspect body_names and contact_body in an episode.h5 file.

Usage:
    python check_contact_h5.py path/to/episode.h5
"""

import logging
import sys
from pathlib import Path

import h5py
import numpy as np
from tabulate import tabulate

logger = logging.getLogger(__name__)


def main(h5_path: Path) -> None:
    with h5py.File(h5_path, "r") as f:
        # Static legend
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["body_names"][:]]
        logger.info("\n=== body_names (index → name) ===")
        logger.info(tabulate(enumerate(names), headers=["ID", "name"]))

        # Contact bodies per time-step
        contact_body = f["contact_body"][:]  # ragged VLEN
        ncon = f["contact_count"][:]

        logger.info("\n=== quick check: first 10 timesteps ===")
        rows = []
        for i in range(min(10, len(contact_body))):
            ids = contact_body[i].reshape(-1, 2) if contact_body[i].size else np.empty((0, 2), dtype=int)
            if ids.size:
                pairs = ", ".join(f"{a}/{b} ({names[a]} / {names[b]})" for a, b in ids)
            else:
                pairs = "—"
            rows.append([i, ncon[i], pairs])
        logger.info(tabulate(rows, headers=["step", "#contacts", "body IDs (names)"]))

        # Optional: aggregate Σ|F| per body across the whole run
        if "contact_force_mag" in f:
            total_force_per_body = np.zeros(len(names))
            wrenches = f["contact_wrench"][:]  # (T,) ragged
            for body_pairs, flat in zip(contact_body, wrenches):
                if body_pairs.size == 0:
                    continue
                forces = flat.reshape(-1, 6)[:, :3]  # Fx,Fy,Fz
                mags = np.linalg.norm(forces, axis=1)
                # body_pairs is now [id1, id2, id3, id4, ...] for each contact pair
                ids = body_pairs.reshape(-1, 2)  # reshape to (n_contacts, 2)
                for (bid1, bid2), mag in zip(ids, mags):
                    # credit the same magnitude to *both* involved bodies
                    total_force_per_body[int(bid1)] += mag
                    total_force_per_body[int(bid2)] += mag

            non_zero = [(names[i], total_force_per_body[i]) for i in np.nonzero(total_force_per_body)[0]]
            logger.info("\n=== Σ|F| per body (whole run) ===")
            logger.info(tabulate(non_zero, headers=["body", "Σ|F| [N]"], floatfmt=".2f"))
    logger.info("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python check_contact_h5.py path/to/episode.h5")
        sys.exit(1)
    main(Path(sys.argv[1]))
