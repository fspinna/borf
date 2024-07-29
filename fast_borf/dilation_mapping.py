import awkward as ak
import numpy as np
from typing import Union


def dilate_panel(
    panel: Union[ak.Array, np.ndarray], d
):  # TODO: implement also the stride
    for i in range(0, d):
        yield panel[:, :, i::d]


if __name__ == "__main__":
    import numpy as np

    panel = [[np.arange(5), np.arange(10)], [np.arange(100), np.arange(7)]]
    panel = ak.Array(panel)
