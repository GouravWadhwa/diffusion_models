class RunningStats:
    def __init__(self, params=[]):
        self.params = {
            param: {
                "average": 0,
                "count": 0
            } for param in params
        }

    def step(self, param, value, count):
        assert (count > 0), "count must be greater than 0"

        if param not in self.params:
            self.params[param] = {
                "average": value,
                "count": 1
            }
        else:
            cur_avg = self.params[param]["average"]
            cur_count = self.params[param]["count"]

            self.params[param]["average"] = cur_avg * (cur_count / (count + cur_count)) + value * (count / (count + cur_count))
    
    def get_averge(self, param):
        if param not in self.params:
            return 0, 0
        else:
            return self.params[param]["average"], self.params[param]["count"]