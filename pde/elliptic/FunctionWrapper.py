class FunctionWrapper:
    def __init__(self, f, is_u):
        # is_u = true: u
        # is_u = false: u'

        self.f = f
        self.is_u = is_u
