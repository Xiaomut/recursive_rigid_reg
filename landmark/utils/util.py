import torch
from functools import reduce
from operator import mul


def flat_softmax(inp):
    """Compute the softmax with all but the first two tensor dimensions combined."""

    orig_size = inp.size()
    flat = inp.view(-1, reduce(mul, orig_size[2:]))
    flat = torch.nn.functional.softmax(flat, -1)
    return flat.view(*orig_size)


def linear_expectation(probs, values):
    assert (len(values) == probs.ndimension() - 2)
    expectation = []
    for i in range(2, probs.ndimension()):
        # Marginalise probabilities
        marg = probs
        for j in range(probs.ndimension() - 1, 1, -1):
            if i != j:
                marg = marg.sum(j, keepdim=False)
        # Calculate expectation along axis `i`
        expectation.append(
            (marg * values[len(expectation)]).sum(-1, keepdim=False))
    return torch.stack(expectation, -1)


def normalized_linspace(length, dtype=None, device=None):
    """Generate a vector with values ranging from -1 to 1.

    Note that the values correspond to the "centre" of each cell, so
    -1 and 1 are always conceptually outside the bounds of the vector.
    For example, if length = 4, the following vector is generated:

    ```text
     [ -0.75, -0.25,  0.25,  0.75 ]
     ^              ^             ^
    -1              0             1
    ```

    Args:
        length: The length of the vector

    Returns:
        The generated vector
    """
    if isinstance(length, torch.Tensor):
        length = length.to(device, dtype)
    first = -(length - 1.0) / length
    return torch.arange(length, dtype=dtype,
                        device=device) * (2.0 / length) + first


def soft_argmax(heatmaps, normalized_coordinates=True):
    if normalized_coordinates:
        values = [
            normalized_linspace(d,
                                dtype=heatmaps.dtype,
                                device=heatmaps.device)
            for d in heatmaps.size()[2:]
        ]
    else:
        values = [
            torch.arange(0, d, dtype=heatmaps.dtype, device=heatmaps.device)
            for d in heatmaps.size()[2:]
        ]
    coords = linear_expectation(heatmaps, values)
    # We flip the tensor like this instead of using `coords.flip(-1)` because aten::flip is not yet
    # supported by the ONNX exporter.
    coords = torch.cat(tuple(reversed(coords.split(1, -1))), -1)
    return coords


def dsnt(heatmaps, **kwargs):
    """Differentiable spatial to numerical transform.

    Args:
        heatmaps (torch.Tensor): Spatial representation of locations

    Returns:
        Numerical coordinates corresponding to the locations in the heatmaps.
    """
    return soft_argmax(heatmaps, **kwargs)