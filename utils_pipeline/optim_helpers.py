"""Utilities for optimization."""

import datetime
from typing import Any


def print_info_step(info_step: dict[str, Any], format_types: list[str], print_header: bool = False) -> None:
    """Print information recorded in info steps with the formatting given in format_types.
    
    Args:
      info_step: info recorded at the given optimization step such as dict(train_loss=..., test_acc=...,
        iteration=...)
      format_types: formatting for each of the entries such as ['scientific', 'float', 'int']
      print_header: whether to print the header (i.e. the keys of info_step)
    """
    assert len(format_types) == len(info_step)
    keys, values = list(info_step.keys()), list(info_step.values())
    if print_header:
        print(('{:<20s}'*len(keys)).format(*keys))
    to_print = ''
    for value, format_type in zip(info_step.values(), format_types):
        if format_type == 'int':
            to_print = to_print + '{:<20d}'.format(value)
        elif format_type == 'time':
            # to_print = to_print + '{:<20s}'.format(str(datetime.timedelta(seconds=value)).split('.')[0])
            to_print = to_print + '{:<20s}'.format(str(datetime.timedelta(seconds=value)))
        elif format_type == 'float':
            to_print = to_print + '{:<20.2f}'.format(value)
        elif format_type == 'scientific':
            to_print = to_print + '{:<20.2e}'.format(value)
        elif format_type == 'string':
            to_print = to_print + '{:<20s}'.format(value)
        else:
            raise NotImplementedError
    print(to_print)
