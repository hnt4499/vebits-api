def get_classes(class_to_be_detected, labelmap_dict):
    if class_to_be_detected == "all":
        return "all"
    else:
        return [labelmap_dict[item] for item in class_to_be_detected.split(',')]

def assert_type(objects, types, raise_error=False):
    """Assert object(s) from allowed types (instances).

    Parameters
    ----------
    objects :
        Single object or a list of objects
    types :
        Single type or a list of types.
    raise_error :
        Whether to raise error.
    Return:
    ----------
    Return boolean value if raise_error=False.
    """

    if not isinstance(objects, list):
        objects = [objects]
    if not isinstance(types, list):
        types = [types]

    for object in objects:
        if not any([isinstance(object, type) for type in types]):
            if raise_error:
                raise ValueError("Invalid input data type. List of allowed types: {}. "
                                 "Got {} instead".format(str(types)[1:-1],
                                                          type(object)))
            else:
                return False
    return True


def convert(object, convert_function, expected_type):
    if not isinstance(object, expected_type):
        try:
            return convert_function(object)
        except:
            raise TypeError("Cannot convert {} to {}".format(type(object),
                                                             expected_type))
    else:
        return object


def raise_type_error(true_type, expected_type):
    if not isinstance(expected_type, list) or len(expected_type) == 1:
        raise TypeError("Invalid input data type. Expected {}. "
                        "Got {} instead".format(expected_type, true_type))
    else:
        raise TypeError("Invalid input data type. Expected one of the following:"
                        " {}. Got {} instead".format(expected_type, true_type))
