from typing import Any

from .nav_technique import NavTechnique


class NavTechniqueStars(NavTechnique):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(logger_name='NavTechniqueStars', **kwargs)

    def navigate(self):
        pass
