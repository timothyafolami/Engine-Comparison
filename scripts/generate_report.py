#!/usr/bin/env python3
"""
Deprecated wrapper: consolidated reporting now lives in ve.reporting.
This script forwards to ve.reporting.write_final_report_markdown for compatibility.
"""
from __future__ import annotations

from pathlib import Path

from ve.reporting import write_final_report_markdown


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "output"
    print("[generate_report] Using consolidated reporter: ve.reporting.write_final_report_markdown")
    write_final_report_markdown(out_dir)


if __name__ == "__main__":
    main()

