"""Types of the project
"""
from typing import (
    Mapping, List, Tuple, Union
    )

SliceAnnotation = Mapping[
        str,
        Mapping[
            str,
            Union[
                List[Tuple[float, float]],
                float,
                Tuple[float, float]
                ]
            ]
    ]
