class ModelError(Exception):
    def __init__(self) -> None:
        super().__init__('Models dosenot contain equal number of atoms')
class AAError(Exception):
    def __init__(self) -> None:
        super().__init__('PDB file contains other then amino acids')
class MissingAtom(Exception):
    def __init__(self,value) -> None:
        super().__init__(' '.join(value))
