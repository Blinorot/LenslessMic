def get_roi_indexes(top_left, height, width, n_dim=4, axis=(-3, -2)):
    """
    Get indexes corresponding to region of interest (ROI).

    Args:
        top_left (tuple[int, int]): top left corner indexes.
        height (int): height of an object.
        width (int): width of an object.
        n_dim (int): number of dimensions.
        axis (tuple[int, int]): tuple with axis index for H and W.
    Returns:
        indexes (tuple[slice, slice, ...]): indexes of ROI.
            Slices for each dimension.
    """
    # extract according to axis
    index = [slice(None)] * n_dim
    index[axis[0]] = slice(top_left[0], top_left[0] + height)
    index[axis[1]] = slice(top_left[1], top_left[1] + width)

    return tuple(index)
