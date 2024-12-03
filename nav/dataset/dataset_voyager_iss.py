class DataSetVoyagerISS:
    def __init__(self):
        ...

    def yield_observations(self):
        ...

    @staticmethod
    def image_name_valid(name: str) -> bool:
        """True if an image name is valid for this instrument."""

        name = name.upper()

        # Cddddddd
        if len(name) != 8 or name[0] != 'C':
            return False

        try:
            _ = int(name[1:])
        except ValueError:
            return False

        return True
