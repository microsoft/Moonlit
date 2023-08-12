from typing import Callable, List, Union, Optional
from modules.search_space.macro_arch.macro_arch import MacroArch
from modules.search_space.superspace import SuperSpace, get_superspace


class SupernetConfig:

    def __init__(self, superspace: Union[str, SuperSpace], building_block_choices: Union[str, List[int]], width_window_choices: Optional[Union[str, List[int]]]=None) -> None:
        if isinstance(superspace, str):
            superspace = get_superspace(superspace)
        self.superspace = superspace
        self.width_window_choices = self.superspace.get_padded_width_window_choices(width_window_choices, padded=False)
        self.building_blocks_choices = SuperSpace.str_to_list(building_block_choices)

    def __str__(self) -> str:
        return f'{self.superspace.NAME}-{SuperSpace.list_to_str(self.building_blocks_choices)}-{SuperSpace.list_to_str(self.width_window_choices)}'
        