def rescale(val,val_scale,new_scale):
    """Rescale value from its original scale to a new scale

    :param val: The value to be rescaled
    :param val_scale: two element list or tuple with original scale
    :param new_scale: two element list or tuple with new scale
    :returns: Rescaled value
    :rtype: float
    """
    return (val-val_scale[0])/float(val_scale[1]-val_scale[0]) * (new_scale[1]-new_scale[0]) + new_scale[0]
    
