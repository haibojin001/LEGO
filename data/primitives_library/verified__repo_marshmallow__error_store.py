from marshmallow.exceptions import SCHEMA


class ErrorStore:
    def __init__(self):
        self.errors = {}

    def store_error(self, messages, field_name=SCHEMA, index=None):
        messages = copy_containers(messages)
        if field_name != SCHEMA or not isinstance(messages, dict):
            messages = {field_name: messages}
        if index is not None:
            messages = {index: messages}
        self.errors = merge_errors(self.errors, messages)


def copy_containers(errors):
    if isinstance(errors, list):
        return [copy_containers(value) for value in errors]
    if isinstance(errors, dict):
        return {key: copy_containers(value) for key, value in errors.items()}
    return errors


def merge_errors(errors1, errors2):
    """Deeply merge two error messages.

    The format of ``errors1`` and ``errors2`` matches the ``message``
    parameter of :exc:`marshmallow.exceptions.ValidationError`.
    """
    if not errors1:
        return errors2
    if not errors2:
        return errors1

    if isinstance(errors1, list):
        if isinstance(errors2, list):
            errors1.extend(errors2)
            return errors1
        if isinstance(errors2, dict):
            errors2[SCHEMA] = merge_errors(errors1, errors2.get(SCHEMA))
            return errors2
        errors1.append(errors2)
        return errors1

    if isinstance(errors1, dict):
        if isinstance(errors2, dict):
            for key, value in errors2.items():
                if key in errors1:
                    errors1[key] = merge_errors(errors1[key], value)
                else:
                    errors1[key] = value
            return errors1
        errors1[SCHEMA] = merge_errors(errors1.get(SCHEMA), errors2)
        return errors1

    if isinstance(errors2, list):
        return [errors1, *errors2]
    if isinstance(errors2, dict):
        errors2[SCHEMA] = merge_errors(errors1, errors2.get(SCHEMA))
        return errors2
    return [errors1, errors2]