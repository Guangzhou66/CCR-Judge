from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_mmlu_ori_protocol import (
    PROJECT_ROOT,
    get_kwargs,
    main as _ori_main,
    parse_args as _ori_parse_args,
)


DEPRECATION_NOTE = (
    "Compatibility wrapper: experiments/run_mmlu.py now forwards to "
    "experiments/run_mmlu_ori_protocol.py. "
    "Use run_mmlu_ori_protocol.py for the canonical online protocol path."
)


def parse_args():
    return _ori_parse_args()


async def main() -> None:
    await _ori_main()


__all__ = [
    "DEPRECATION_NOTE",
    "PROJECT_ROOT",
    "get_kwargs",
    "parse_args",
    "main",
]


if __name__ == "__main__":
    asyncio.run(main())
