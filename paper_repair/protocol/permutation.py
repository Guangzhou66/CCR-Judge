from __future__ import annotations

import random
from typing import List

from paper_repair.repair.ccr_types import FrozenCandidateRecord


def candidate_permutation(
    record: FrozenCandidateRecord,
    *,
    judge_shuffle: bool,
    shuffle_seed: int,
) -> List[str]:
    agent_ids = [candidate.agent_id for candidate in record.candidates]
    if not judge_shuffle:
        return agent_ids
    rng = random.Random(int(shuffle_seed) + (int(record.record_index) * 9973))
    permuted = agent_ids.copy()
    rng.shuffle(permuted)
    return permuted
