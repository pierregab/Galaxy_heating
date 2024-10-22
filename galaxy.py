from system import System



class Galaxy(System):
    '''
    Galaxy class :
    @method
        - addStars
        - setMass
        - seta
        - setb
    '''
    def __init__(self, mass:float=1e10, semiMajorAxis:float=2.5, semiMinorAxis:float=2.5/20, *args, **kwargs) -> None:
        if type(semiMajorAxis) != float:
            try:
                semiMajorAxis = float(semiMajorAxis)
            except ValueError:
                raise ValueError(f"Parameter `semiMajorAxis` should be convertible to float type (currently {type(semiMajorAxis)}).")
        if type(semiMinorAxis) != float:
            try:
                semiMinorAxis = float(semiMinorAxis)
            except ValueError:
                raise ValueError(f"Parameter `semiMinorAxis` should be convertible to float type (currently {type(semiMinorAxis)}).")
        self.a = semiMajorAxis
        self.b = semiMinorAxis
        super().__init__(self, mass)
        pass

