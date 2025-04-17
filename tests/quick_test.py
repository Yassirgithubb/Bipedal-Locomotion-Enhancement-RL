from dataclasses import dataclass, field
from legged_gym.utils import configclass


@configclass
class A:
    defg: int = 2
    abc = 1


a = A()
print(A.__dict__)
print(a.__dict__)
a = A()


class B(A):
    defg = 2
