"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
# import additional ...
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, DotProduct, WhiteKernel

# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA

# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        self.data = {
            "x": np.array([]), 
            "f": np.array([]), 
            "v": np.array([])
        }

        # According to the specs in handout
        self.kernel_f = Matern(nu=2.5, length_scale=0.011) + WhiteKernel(0.15**2) #length_scale_bounds=(0.001, 10)) #ConstantKernel(constant_value=0.5, constant_value_bounds=(0.1, 1)) * RBF(length_scale=0.5, length_scale_bounds=(0.01, 0.25)) # Matern(nu=2.5, length_scale_bounds=(0.5, 10))
        self.kernel_v = DotProduct(sigma_0=1.4) + Matern(nu=2.5, length_scale=0.011) + WhiteKernel(0.001**2) #sigma_0=0.0001, sigma_0_bounds=(0.0001, 0.01)) + Matern(nu=2.5, length_scale_bounds=(0.01, 10)) #DotProduct(sigma_0=0, sigma_0_bounds="fixed") + ConstantKernel(constant_value=2**0.5, constant_value_bounds=(0.1, 1)) * RBF(length_scale=0.5, length_scale_bounds=(0.01, 0.25)) # DotProduct() + Matern(nu=2.5, length_scale_bounds=(0.5, 10))
        self.f = GaussianProcessRegressor(kernel=self.kernel_f) 
        self.v = GaussianProcessRegressor(kernel=self.kernel_v)
        self.prev_optimum_f = 0
        self.prev_optimum_v = 0
        
        self.prior_mean_v = 4

        # For the Upper Confidence Bound acquisition function
        self.beta = 0.49
        # For the Lagrangian Relaxation criterion
        # self.lambda_ = 0.8 # => passes medium baseline with Expected Improvement
        self.lambda_ = 1000 # => passes hard baseline on Upper Confidence Bound

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        self.f.fit(X=self.data["x"].reshape(-1, 1), y=self.data["f"])
        self.v.fit(X=self.data["x"].reshape(-1, 1), y=self.data["v"])

        return self.optimize_acquisition_function()
        #raise NotImplementedError

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.

        # Upper Confidence Bound acquisition function
        def upperConfidenceBound(f: GaussianProcessRegressor, x: np.ndarray, beta: float = self.beta):
            means, stds = f.predict(x, return_std=True)
            return means + beta*stds
        
        ucb_f = upperConfidenceBound(self.f, x)
        return ucb_f - self.lambda_ * np.maximum(self.v.predict(x), np.zeros(x.shape[0]))
    
        # def expectedImprovement(f: GaussianProcessRegressor, x: np.ndarray, prev_optimum: float):
        #     # Select the safest values of x          
        #     means, stds = f.predict(x, return_std=True)
        #     res = means - prev_optimum
        #     arg = res / stds
        #     with np.errstate(divide='warn'):
        #         ei = res * norm.cdf(arg) + stds * norm.pdf(arg)
        #         ei[stds == 0] = 0.0
        #     return ei

        # return expectedImprovement(self.f, x, self.prev_optimum_f) - self.lambda_ * np.maximum(self.v.predict(x), np.zeros(x.shape[0]))

        # ei_f = expectedImprovement(self.f, 0, x, prev_optimum=self.prev_optimum_f)
        # ei_v = expectedImprovement(self.v, self.prior_mean_v, x, prev_optimum=self.prev_optimum_v, safety_threshold=SAFETY_THRESHOLD)

        # Prob that constraint is satisfied:
        # return ei_f - self.lambda_ * ei_v

        # v_means, v_stds = self.v.predict(x, return_std=True)
        # constraint_prob = norm.cdf((x - v_means) / v_stds).reshape(1, )

        # return ei_f * constraint_prob if constraint_prob <= self.prob_threshold else ei_f

        # # Probability of Improvement acquisition function
        # def probabilityImprovement(f: GaussianProcessRegressor, x: np.ndarray, prev_optimum: float):
        #     means, stds = f.predict(x, return_std=True)
        #     return norm.cdf((means - prev_optimum) / stds)

        # pi_f = probabilityImprovement(self.f, x, self.prev_optimum_f)
        # pi_v = probabilityImprovement(self.v, x, np.max([self.prev_optimum_v, SAFETY_THRESHOLD]))

        # return pi_f - 200 * pi_v

        #raise NotImplementedError

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        self.data["x"] = np.append(self.data["x"], x)
        self.data["f"] = np.append(self.data["f"], f)
        self.data["v"] = np.append(self.data["v"], v)
        #raise NotImplementedError

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        self.data["f"][self.data["v"] >= SAFETY_THRESHOLD] = -np.inf # Penalize the images of f if the constraint is not fulfilled
        ind = np.argmax(self.data["f"])
        self.prev_optimum_f = self.data["f"][ind]
        self.prev_optimum_v = self.data["v"][ind]

        return self.data["x"][ind]
        #raise NotImplementedError

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # # Check for valid shape
        # assert x.shape == (1, DOMAIN.shape[0]), \
        #     f"The function next recommendation must return a numpy array of " \
        #     f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
