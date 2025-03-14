class Dual:
    """
    A class that implementes hyperreal dual numbers.
    These are numbers of the form a + b*ε,
    where a,b are real numbers and ε^2 = 0.
    Any analytic function f(x) has the property that
    f(a + b*ε) = f(a) + b*f'(a)*ε.
    """

    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def __add__(self, other: "Dual"):
        return Dual(self.a + other.a, self.b + other.b)

    def __sub__(self, other: "Dual"):
        return Dual(self.a - other.a, self.b - other.b)

    def __mul__(self, other: "Dual"):
        return Dual(self.a * other.a, self.a * other.b + self.b * other.a)

    def __pow__(self, beta: float):
        return Dual(self.a**beta, beta * self.b * (self.a**beta - 1.0))
