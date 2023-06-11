import numpy as np
import pytest

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from production.scripts import log_transformer


@pytest.mark.parametrize("test_input, expected", [(100, np.log(1 + 100))])
def test_log_transformer(test_input, expected):
    assert log_transformer(test_input) == expected
