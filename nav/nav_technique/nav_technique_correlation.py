from typing import Any

from .nav_technique import NavTechnique


class NavTechniqueCorrelation(NavTechnique):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(**kwargs)
