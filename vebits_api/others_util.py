from functools import wraps


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


def import_wrapper(import_function, package_name):
    """
    This function wraps up other functions as decorator and check whether the
    provided package can be imported or not. This should be used to catch any
    ImportError beforehand for functions that require specific packages.
    """
    def wrapper(function):
        @wraps(function)
        def _wrapper(*args, **kwargs):
            try:
                import_function()
            except ImportError:
                print("Cannot import package {}".format(package_name))
            return function(*args, **kwargs)
        return _wrapper
    return wrapper


def check_import(is_imported, package_name):
    """
    This function wraps up other functions as decorator and check whether the
    `package_name` has been imported, given a boolean value.
    """
    def wrapper(function):
        @wraps(function)
        def _wrapper(*args, **kwargs):
            for x, y in zip(is_imported, package_name):
                if not x:
                    raise ImportError("Module {} is not yet "
                                      "imported.".format(y))
            return function(*args, **kwargs)
        return _wrapper
    return wrapper


def get_classes(class_to_be_detected, labelmap_dict):
    """
    This function is used to process `class to be detected` argument
    passed in some scripts.
    """
    if class_to_be_detected == "all":
        return "all"
    else:
        return [labelmap_dict[item] for item in class_to_be_detected.split(',')]


def get_batch_names(img_name, num):
    """
    This function is used to get images' name by batch for a single
    original image, e.g., 1.jpg ---> 1_0.jpg, 1_1.jpg,...
    """
    name, ext = os.path.splitext(img_name)
    return ["{}_{}{}".format(name, i, ext) for i in range(num)]
