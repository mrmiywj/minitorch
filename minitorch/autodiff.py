from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Dict

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    vals_plus = [x for x in vals]
    vals_plus[arg] += epsilon
    return (f(*vals_plus) - f(*vals)) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    queue:Iterable[Variable] = []
    ret = []
    id2var = {}
    id2cnt = {}
    queue.append(variable)
    while queue:
        now = queue.pop()
        if now.unique_id in id2var:
            continue
        id2var[now.unique_id] =  now
        if now.unique_id not in id2cnt:
            id2cnt[now.unique_id] = 0
        for p in now.parents:
            if p.is_constant():
                continue
            if p.unique_id not in id2cnt:
                id2cnt[p.unique_id] = 0
            id2cnt[p.unique_id] += 1
            queue.append(p)
    for i, cnt in id2cnt.items():
        if cnt == 0:
            queue.append(i)
    while queue:
        nowid = queue.pop()
        now = id2var[nowid]
        ret.append(now)
        for p in now.parents:
            if p.is_constant():
                continue
            id2cnt[p.unique_id] -= 1
            if id2cnt[p.unique_id] == 0:
                queue.append(p.unique_id)
    return ret



def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    vars = topological_sort(variable)
    id2var:Dict[int, Variable] = dict()
    id2deri:Dict[int, float] = dict()
    for v in vars:
        id2var[v.unique_id] = v
        id2deri[v.unique_id] = 0.0
    id2deri[variable.unique_id] = deriv
    for v in vars:
        if v.is_leaf():
            continue
        tuples = v.chain_rule(id2deri[v.unique_id])
        for (parent, deriv_p ) in tuples:
            if parent.is_leaf():
                parent.accumulate_derivative(deriv_p)
            else: 
                id2deri[parent.unique_id] += deriv_p

        


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
