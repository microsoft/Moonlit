from typing import List

def channel_list(left: int, right: int, step: int=8) -> List[int]:
    return list(range(left, right + 1, step))