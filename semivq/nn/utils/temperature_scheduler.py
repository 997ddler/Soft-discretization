import numpy as np
class temperature_scheduler_base():
    def __init__(self, temperature=1.0):
        self.temperature = temperature
        self.iter = 0
    def calc_temperature(self):
        raise NotImplementedError

class scheduler_increase_linear(temperature_scheduler_base):
    def __init__(self, gain = 1e-5, temperature=0.01):
        super().__init__(temperature)
        self.gain = gain

    def calc_temperature(self):
        self.temperature = np.min([self.temperature + self.iter * self.gain, 1.0])
        self.iter = self.iter + 1
        return self.temperature

class scheduler_increase_exp(temperature_scheduler_base):

    def __init__(self, temperature=0.0):
        super().__init__(temperature)

    def calc_temperature(self):
        self.temperature = 1 - np.exp(-self.iter/100)
        self.iter = self.iter + 1
        return self.temperature

class scheduler_increase_inverse(temperature_scheduler_base):
    def __init__(self, temperature=0.0):
        super().__init__(temperature)

    def calc_temperature(self):
        self.temperature = 1 - 1.0 / (1 + self.iter)
        self.iter = self.iter + 1
        return self.temperature


class scheduler_decrease(temperature_scheduler_base):

    def __init__(self, gain=1e-5, temperature=5.0):
        super().__init__(temperature)
        self.gain = gain

    def calc_temperature(self):
        self.temperature = np.max([self.temperature - self.iter * self.gain, 1.0])
        self.iter = self.iter + 1
        return self.temperature