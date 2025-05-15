class Polynomial:
    """
    Represents polynomials, to be used as values for parametric models.
    Polynomials are represented as an dictionary with n-dimensional tuples as keys.

    Args:
        coefficients: coefficients of the terms
    """

    coefficients: dict[tuple,float]

    def __init__(self):
        self.coefficients = {}

    def set_coefficient(self, exponents: tuple[int, ...], coefficient: float):
        if self.coefficients != {}:
            print(self.coefficients)
            length = len(list(self.coefficients.keys())[0])
            if length != len(exponents):
                raise RuntimeError(f"The length of the exponents tuple should be: {length}")
        self.coefficients[exponents] = coefficient

    def get_dimension(self) -> int:
        if self.coefficients is not {}:
            return len(list(self.coefficients.keys())[0])
        else:
            return 0

    def get_degree(self) -> int:
        if self.coefficients is not {}:
            largest = 0
            for term in self.coefficients.keys():
                for exponent in term:
                    if exponent > largest:
                        largest = exponent

            return largest
        else:
            return 0

    # TODO valuation function

    def __str__(self) -> str:
        s = ""
        # we iterate through each term
        for exponents, coefficient in self.coefficients.items():
            # we only print terms with nonzero coefficients
            if coefficient != 0:
                # we don't print coefficients that are 1
                if coefficient != 1:
                    s += f"{coefficient}*"

                # we print the variables with their corresponding powers
                # if the tuple only consists of zeroes then we are left with 1
                all_zero = True
                for variable,exponent in enumerate(exponents):
                    if exponent != 0:
                        all_zero = False
                        s += f"x_{variable}"
                        if exponent != 1:
                            s += f"^{exponent}"
                if all_zero:
                    s += "1"
                s += " + "

        return s[:-3]


class RationalFunction:
    """
    Represents rational functions, to be used as values for parametric models
    Rational functions are represented as a pair of polynomials.

     Args:
        numerator: Polynomial in the numerator
        denominator: Polynomial in the denominator
    """

    numerator: Polynomial
    denominator: Polynomial

    def __init__(self, numerator: Polynomial, denominator: Polynomial):
        denominator_all_zero = True
        for exponents, coefficient in denominator.coefficients.items():
            if coefficient != 0:
                denominator_all_zero = False

        if not denominator_all_zero:
            self.numerator = numerator
            self.denominator = denominator
        else:
            raise RuntimeError("dividing by 0 is not allowed")


    def get_dimension(self) -> int:
        if numerator.dimension > denominator.dimension:
            return numerator.dimension
        else:
            return denominator.dimension

    # TODO valuation function

    def __str__(self) -> str:
        s = str(self.numerator) + "\n"
        # the length of the division line depends on the length of the largest polynomial
        for i in range(max(len(str(self.numerator)), len(str(self.denominator)))):
            s += "-"
        s += "\n" + str(self.denominator)
        return s


Parametric = Polynomial | RationalFunction

if __name__ == "__main__":
    polynomial1 = Polynomial()
    polynomial2 = Polynomial()
    polynomial1.set_coefficient((0, 0, 1), 5)
    polynomial1.set_coefficient((3, 2, 1), 2.4)
    polynomial2.set_coefficient((2, 3, 2, 1), 1)
    polynomial2.set_coefficient((0, 0, 0, 0), 3)

    rationalfunction = RationalFunction(polynomial1, polynomial2)

    print(rationalfunction)
