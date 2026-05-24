# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/21 16:09
Create User : 19410
Desc : xxx
"""

from typing import Union, Dict, Tuple

from langchain_core.tools import Tool


class SupportDictArgsTool(Tool):
    def _to_args_and_kwargs(self, tool_input: Union[str, Dict], tool_call_id) -> Tuple[Tuple, Dict]:
        """Convert tool input to pydantic model."""
        if isinstance(tool_input, str):
            return (tool_input,), {}
        else:
            return (), tool_input
