""" Contains different kinds of form validators.
"""
from wtforms.validators import ValidationError


class CheckNumberRange(object):
    def __init__(self, lower, upper, hint=""):
        self.lower = lower
        self.upper = upper
        self.hint = hint

    def __call__(self, form, field):

        if field.data is not None:
            if not (self.lower <= float(field.data) <= self.upper):
                if self.hint:
                    raise ValidationError(self.hint)
                else:
                    raise ValidationError("Not in correct range")
