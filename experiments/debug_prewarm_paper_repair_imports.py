from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _emit(message: str, debug_log: Path) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with debug_log.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = PROJECT_ROOT / "result" / "paper_repair_builder_debug" / f"prewarm_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_log = out_dir / "debug.log"
    report_path = out_dir / "prewarm_report.json"

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PAPER_REPAIR_DEBUG_LOG"] = str(debug_log)
    sys.path.append(str(PROJECT_ROOT))

    steps = [
        "my_datasets.mmlu_dataset",
        "KVCOMM.prompt.mmlu_prompt_set",
        "KVCOMM.llm.gpt_chat",
        "paper_repair.protocol.frozen_pack",
    ]
    rows = []
    for module_name in steps:
        _emit(f"[PREWARM] before import {module_name}", debug_log)
        start = time.perf_counter()
        status = "ok"
        error = None
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - start
        _emit(f"[PREWARM] after import {module_name} status={status} elapsed={elapsed:.3f}s", debug_log)
        rows.append(
            {
                "module": module_name,
                "status": status,
                "elapsed_sec": elapsed,
                "error": error,
            }
        )
        if status != "ok":
            break

    report_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _emit(f"[PREWARM] report saved path={report_path}", debug_log)


if __name__ == "__main__":
    main()
