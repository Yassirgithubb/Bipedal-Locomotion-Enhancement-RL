import torch

import contextlib
from time import time


@contextlib.contextmanager
def timer(str):

    start = time()
    yield
    print("---")
    print(str, f" took {time()-start}")


def check_is_copy(t1, t2):
    t1[0, 0] = 123
    t2[0, 0] = 345
    if t1[0, 0] != t2[0, 0]:
        print("copy")
    else:
        print("no copy")


device = "cuda:0"
# indexing test

t = torch.arange(4000 * 100, device=device).view(4000, 100)
ids_tensor = torch.randint(0, 100, (4000,), device=device)
ids_list = ids_tensor.tolist()
slice_idx = list(range(4000))
a = torch.arange(4000 * 100, device=device).view(4000, 100)
b = torch.rand(4000, 100, device=device)

# print("--------------")
# a = torch.randint(0, 10, (4000, 100), device=device)
# with timer("indexing with tensor"):
#     a = t[ids_tensor]
# check_is_copy(a, t)

# a = torch.randint(0, 10, (4000, 100), device=device)
# with timer("indexing with list"):
#     a = t[ids_list]
# check_is_copy(a, t)

# a = torch.randint(0, 10, (4000, 100), device=device)
# with timer("slicing"):
#     a = t[:1000]
# check_is_copy(a, t)

# a = torch.randint(0, 10, (4000, 100), device=device)
# with timer("indexed slicing"):
#     a = t[slice_idx]
# check_is_copy(a, t)

# a = torch.randint(0, 10, (4000, 100), device=device)
# with timer("[...] indexing"):
#     a = t[...]  # no copy
# check_is_copy(a, t)

# a = torch.randint(0, 10, (4000, 100), device=device)
# with timer("[:] indexing"):
#     a = t[:]
# check_is_copy(a, t)

# a = torch.randint(0, 10, (4000, 100), device=device)
# with timer("[...] = [...] indexing"):
#     a[...] = t[...]  # no copy
# check_is_copy(a, t)

# a = torch.randint(0, 10, (4000, 100), device=device)
# with timer("[:] = [:] indexing"):
#     a[:] = t[:]
# check_is_copy(a, t)

# a = torch.randint(0, 10, (4000, 100), device=device)
# with timer("cloning"):
#     a = t.clone()
# check_is_copy(a, t)
# print("--------------")

with timer("torch.rand"):
    c = torch.rand_like(b)

with timer("torch.rand + id_list"):
    c[ids_list] = torch.rand_like(b[ids_list])

with timer("torch.rand + id_tensor"):
    c[ids_tensor] = torch.rand_like(b[ids_tensor])

with timer("uniform"):
    b.uniform_(0.0, 1.0)
