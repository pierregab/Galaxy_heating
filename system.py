

class System:
    '''
    System class :
    @class.attribute
        - lengthScale
        - massScale
        - timeScale
        - vScale
        - plotEnergyError
        - plotAngularMomentumError
        - plotTrajectories
        - plotExecutionTime
    @method
        - leapfrog
        - RK4
        - getPotential
        - update
    '''
    def __init__(self, systemSubClassInstance, mass:float, *args, **kwargs) -> None:
        if type(mass) != float:
            try:
                mass = float(mass)
            except ValueError:
                raise ValueError(f"Parameter `mass` should be convertible to float type (currently {type(mass)}).")
        systemSubClassInstance.mass = 1
        pass

