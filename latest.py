import numpy as np
import matplotlib.pyplot as plt

class AdamsIntegrator:
    """
    A python class to perform ABM method to integrate ODE
    """

    def __init__(self, n_steps, start, stop):
        self.n_steps, self.start, self.stop = n_steps, start, stop
        self.h_step = (stop - start) / n_steps
        self.h_step2 = self.h_step / n_steps
        self.independent_var = np.linspace(start, stop, n_steps + 1)
        self.dependent_var = np.zeros_like(self.independent_var)

    def adams_function(self, t, y):
        """
        Returns function of an initial value problem
        """
        return (t - y) / 2

    def runge_kutta(self, t, y):
        """
        Returns dependent variable from runge kutta method of order 4
        """
        k0 = self.h_step * self.adams_function(t, y)
        k1 = self.h_step * self.adams_function(t + self.h_step / 2, y + k0 / 2)
        k2 = self.h_step * self.adams_function(t + self.h_step / 2, y + k1 / 2)
        k3 = self.h_step * self.adams_function(t + self.h_step, y + k2)
        return y + (1 / 6) * (k0 + 2 * k1 + 2 * k2 + k3)

    def adams_bashforth_predictor(self):
        self.dependent_var[0] = 1
        for k in range(1, 4):
            self.dependent_var[k] = self.runge_kutta(self.independent_var[k - 1], self.dependent_var[k - 1])
        for k in range(3, self.n_steps):
            p = self.dependent_var[k] + self.h_step2 * (
                    -9 * self.adams_function(self.independent_var[k], self.dependent_var[k])
                    + 37 * self.adams_function(self.independent_var[k - 1], self.dependent_var[k - 1])
                    - 59 * self.adams_function(self.independent_var[k - 2], self.dependent_var[k - 2])
                    + 55 * self.adams_function(self.independent_var[k - 3], self.dependent_var[k - 3])
            )
            self.independent_var[k + 1] = self.start + self.h_step * (k + 1)
            self.dependent_var[k + 1] = self.runge_kutta(self.independent_var[k], self.dependent_var[k])

    def get_final_result(self):
        """
        Generate and plot final results
        """
        self.adams_bashforth_predictor()
        ysol = 3 * np.exp(-self.independent_var / 2) - 2 + self.independent_var

        print("{: >3} {: >4} {: >15} {: >10}".format("k", "t", "Y numerical", "Y exact"))
        for k in range(self.n_steps + 1):
            print("{: 3d},{: 5.3f},{: 12.11f},{: 12.11f}".format(k, self.independent_var[k], self.dependent_var[k], ysol[k]))
            plt.plot(self.independent_var[: self.n_steps + 1], self.dependent_var[: self.n_steps + 1], "o")
            plt.plot(self.independent_var[: self.n_steps + 1], ysol[: self.n_steps + 1])
        plt.show()


if __name__ == "__main__":
    integrator = AdamsIntegrator(n_steps=24, start=0, stop=3)
    integrator.get_final_result()

