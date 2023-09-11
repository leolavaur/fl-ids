"""Module for poisoning-related data structures and helpers."""

import re
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple


class PoisonOp(str, Enum):
    INC = "+"
    DEC = "-"


class PoisonTask(NamedTuple):
    fraction: float
    operation: PoisonOp = PoisonOp.INC


PoisonTasks = Dict[int, Optional[PoisonTask]]


# Version with immutable value
# -----------------------------
class PoisonIns(NamedTuple):
    target: List[str]
    base: PoisonTask
    tasks: Optional[PoisonTasks] = None
    poison_eval: bool = False


#     # ensure that base.operation is Op.INC
#     # TODO: find a way to do this on NamedTuple
#     def __post_init__(self):
#         if self.base.operation != PoisonOp.INC:
#             raise ValueError(
#                 f"PoisonIns.base.operation must be {PoisonOp.INC}, got {self.base.operation}"
#             )

# Version with input validation
# -----------------------------
# class PoisonIns:
#     def __init__(
#         self,
#         target: List[str],
#         base: PoisonTask,
#         tasks: Optional[PoisonTasks] = None,
#         poison_eval: bool = False,
#     ) -> None:
#         self._target = target
#         self._base = base
#         self._tasks = tasks
#         self._poison_eval = poison_eval

#     def __new__(
#         cls,
#         target: List[str],
#         base: PoisonTask,
#         tasks: Optional[PoisonTasks] = None,
#         poison_eval: bool = False,
#     ) -> "PoisonIns":
#         if base.operation != PoisonOp.INC:
#             raise ValueError(
#                 f"PoisonIns.base.operation must be {PoisonOp.INC}, got {base.operation}"
#             )
#         return super().__new__(cls)

#     @property
#     def target(self) -> List[str]:
#         return self._target

#     @property
#     def base(self) -> PoisonTask:
#         return self._base

#     @property
#     def tasks(self) -> Optional[PoisonTasks]:
#         return self._tasks

#     @property
#     def poison_eval(self) -> bool:
#         return self._poison_eval


def parse_poisoning_selector(
    selector: str, n_rounds: int
) -> Tuple[PoisonTask, Optional[PoisonTasks]]:
    """Parse poisoning selector.

    The following selectors are be supported:
        * `*`: select all rounds (eg. "0.5*", half of the dataset is poisoned),
            equivalent to "0.5" and "0.5[:]". This is the default selector.
        * `[m:n]`: select rounds from `m` to `n` (inclusive, eg.
            "0.5[1:4]", half of dataset is poisoned in rounds 1 to 4). Note that
            here, indexes start at 1.
        * `{m,n,...}`: select rounds `m` and `n`, and so on (eg. "0.5{0,2}",
            half of dataset is poisoned in rounds 0 and 2)
    The default selector is "*" (eg. "0.5" is equivalent to "0.5*").

    Implementations SHOULD also support the increment operator `+`, which allow to
    increment the fraction of poisoned data set by `x` each selected round. For
    example, "0.5+0.1[0:2]" poisons 50% of the data in round 0, 60% in round 1, and
    70% in round 2. The starting value is 0 if not specified otherwise.

    Implementations CAN support the `-` operator, which allows to decrement the
    fraction of poisoned data set by `x` each selected round. It is up to the
    implementaion to decide whether to support it or not, and raise an error if it
    is not supported.

    Note:
        Selectors are parsed in order, so subsequent selectors can override previous
        ones. For example, "0.0+0.1-0.1" poisons 0% of the data at initialization, and
        since `-0.1` takes precedence over `+0.1`, the implementation will attemps to
        decrement at each round. It is up to the implementation to decide whether to
        raise an error, warn the user, or fail silently.

    Parameters
    ----------
    selector : str
        Fraction of the data set to poison. Implementations SHOULD support the format
        "(o)x(s)" where `x` is a float between 0 and 1, `s` is a round selector, and `o`
        the optional increment operator.

    n_rounds : int
        Number of rounds in the experiment.

    Returns
    -------
    float
        Base fraction of the data set to poison.
    Optional[PoisonTasks]
        Dictionary of poisoning tasks. Keys are the rounds, values are the poisoning
        tasks. A "task" is a increment/decrement operation to apply to the current
        dataset, ie. poison the dataset by `x` more (or less) than the previous round.
    """
    assert type(selector) == str, "Invalid selector; must be a string"

    base_re = r"(?P<base>\d+(?:\.\d+)?)"

    inc_re = r"(?P<op>[+-])(?P<inc>\d+(?:\.\d+)?)"
    range_re = r"(?P<range>\[(?P<from>\d+)?:(?P<to>\d+)?\])"
    set_re = r"(?P<set>\{(?P<set_items>(?:\d+,?)+)\})"
    star_re = r"(?P<star>\*)"
    selector_re = rf"{inc_re}(?:{star_re}|{range_re}|{set_re})?"

    p = re.compile(base_re)
    m = p.match(selector)
    if m is None:
        raise ValueError(f"Invalid selector: '{selector}'; no base value found")

    base = float(m.group("base"))
    tasks: PoisonTasks = {}

    p = re.compile(selector_re)
    for m in p.finditer(selector):
        if m is None:
            raise ValueError(f"Invalid selector: '{selector}'; no base value found")

        rounds = list(range(1, n_rounds + 1))

        for m in p.finditer(selector):
            _op = m.group("op")
            _inc = m.group("inc")
            if _op is None or _inc is None:
                raise ValueError(f"Invalid selector: '{selector}'; no increment found")

            op = PoisonOp(_op)
            inc = float(_inc)

            if m.group("range") is not None:
                from_ = int(m.group("from")) if m.group("from") is not None else 1
                to = int(m.group("to")) if m.group("to") is not None else n_rounds
                rounds = list(range(from_, to + 1))
                if from_ < 1 or to > n_rounds or from_ > to:
                    raise IndexError(
                        f"Invalid range: '{selector}'; round index out of bounds"
                    )
            elif m.group("set") is not None:
                rounds = [
                    r
                    for r in rounds
                    if r in map(int, [int(r) for r in m.group("set_items").split(",")])
                ]
            # else: all rounds

            for r in rounds:
                tasks[r] = PoisonTask(fraction=inc, operation=op)

    return PoisonTask(base), tasks or None
