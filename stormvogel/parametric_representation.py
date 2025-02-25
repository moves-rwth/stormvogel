import numpy as np


class Parameter:
    name: str

    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return self.name


# Parameter = str


class Polynomial:
    coefficients: np.ndarray = np.array([])

    def __init__(
        self, degree: int, dimension: int, coefficients: np.ndarray | None = None
    ):
        if coefficients is None:
            self.coefficients = np.zeroes((degree + 1,) * dimension)
        else:
            self.coefficients = coefficients

    # def set_coefficient(self, variables: tuple[Parameter, ...], coefficient: float):
    #    self.coefficients[tuple[Parameter, ...]] = coefficient

    def __str__(self) -> str:
        s = ""
        for index, pair in enumerate(self.coefficients.items()):
            term = ""
            if str(pair[1]) != "1":
                term += str(pair[1])
            for var in pair[0]:
                term += str(var)
            s += term
            if index < len(self.coefficients.items()) - 1:
                s += " + "
        return s


class RationalFunction:
    numerator: Polynomial
    denominator: Polynomial

    def __init__(self, numerator: Polynomial, denominator: Polynomial):
        self.numerator = numerator
        self.denominator = denominator

    def __str__(self) -> str:
        s = str(self.numerator) + "\n"
        for i in range(max(len(str(self.numerator)), len(str(self.denominator)))):
            s += "-"
        s += "\n" + str(self.denominator)
        return s


if __name__ == "__main__":
    x = Parameter("x")
    y = Parameter("y")
    polynomial1 = Polynomial(3, 3)
    polynomial2 = Polynomial(4, 4)

    function = RationalFunction(polynomial1, polynomial2)

    print(function)
