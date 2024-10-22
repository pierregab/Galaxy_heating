from system import System



class Perturber(System):
    '''
    Perturber Class :
    @method
        - setMass
        - setPos
    '''
    def __init__(self, mass:float, *args, **kwargs) -> None:
        super.__init__(self, mass)
        pass
