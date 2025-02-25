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

    def __init__(self, degree: int, dimension: int):
        self.coefficients = np.zeros((degree + 1,) * dimension)

    def set_coefficient(self, exponents: tuple[int, ...], coefficient: float):
        self.coefficients[exponents] = coefficient

    def __str__(self) -> str:
        s = ""
        for id in np.ndindex(self.coefficients.shape):
            s += f"Index: {id}, Value: {self.coefficients[id]}"
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

    print(polynomial1)

    # function = RationalFunction(polynomial1, polynomial2)

    # print(function)
