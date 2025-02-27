import numpy as np


class Polynomial:
    coefficients: np.ndarray = np.array([])

    def __init__(self, degree: int, dimension: int):
        self.coefficients = np.zeros((degree + 1,) * dimension)

    def set_coefficient(self, exponents: tuple[int, ...], coefficient: float):
        self.coefficients[exponents] = coefficient

    def __str__(self) -> str:
        s = ""
        for index, id in enumerate(np.ndindex(self.coefficients.shape)):
            if self.coefficients[id] != 0:
                if self.coefficients[id] != 1:
                    s += f"{self.coefficients[id]}*"
                for i in range(len(id)):
                    if id[i] != 0:
                        s += f"x_{i}"
                        if id[i] != 1:
                            s += f"^{id[i]}"
                s += " + "

        return s[:-3]


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
    polynomial1 = Polynomial(3, 3)
    polynomial2 = Polynomial(4, 4)
    polynomial1.set_coefficient((0, 0, 1), 5)
    polynomial1.set_coefficient((3, 2, 1), 2.4)
    polynomial2.set_coefficient((2, 3, 2, 1), 1)

    rationalfunction = RationalFunction(polynomial1, polynomial2)

    print(rationalfunction)
