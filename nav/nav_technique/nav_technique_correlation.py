from typing import Any

from .nav_technique import NavTechnique


class NavTechniqueCorrelation(NavTechnique):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(logger_name='NavTechniqueCorrelation', **kwargs)
