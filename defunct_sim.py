import numpy as np

class Agent:
    def __init__(self, group):
        self.group = group
        self.features = self.sample_features()
        self.budget = self.sample_budget()
        self.cost_function = self.sample_cost_function()
        self.true_label = self.sample_true_label()

    def sample_features(self):
        # Adjusted to provide more realistic distributions and variation
        if self.group == 'advantaged':
            debt_to_income_ratio = np.random.normal(30, 5)  # More favorable
            fico_points = np.random.normal(700, 50)  # Higher score
        else:
            debt_to_income_ratio = np.random.normal(50, 10)  # Less favorable
            fico_points = np.random.normal(600, 50)  # Lower score
        return {'debt_to_income_ratio': debt_to_income_ratio, 'fico_points': fico_points}

    def sample_budget(self):
        # More realistic budget distributions
        if self.group == 'advantaged':
            return np.random.normal(2000, 300)  # Higher budget
        else:
            return np.random.normal(800, 150)  # Lower budget

    def sample_cost_function(self):
        # Cost to modify features is more varied and realistic
        if self.group == 'advantaged':
            return lambda x: 0.5 * x  # Cheaper to change features
        else:
            return lambda x: 1.0 * x  # More expensive to change features

    def sample_true_label(self):
        default_probability = 0.5 - 0.05 * self.features['debt_to_income_ratio']
        return np.random.rand() < default_probability

    def decide_modifications(self, regression_vector):
        best_modification = None
        max_utility_gain = -float('inf')
        current_utility = self.calculate_utility(self.features, regression_vector)
        chosen_delta = 0

        # Exploring a more nuanced decision-making strategy
        for feature in ['debt_to_income_ratio', 'fico_points']:
            potential_changes = np.linspace(-10, 10, num=20) if feature == 'fico_points' else np.linspace(-5, 5, num=10)
            for delta in potential_changes:
                new_features = self.features.copy()
                new_features[feature] += delta
                if self.budget >= self.cost_function(abs(delta)):
                    new_utility = self.calculate_utility(new_features, regression_vector)
                    utility_gain = new_utility - current_utility - self.cost_function(abs(delta))

                    if utility_gain > max_utility_gain:
                        max_utility_gain = utility_gain
                        best_modification = (feature, delta)
                        chosen_delta = delta

        if best_modification:
            self.features[best_modification[0]] += chosen_delta
            self.budget -= self.cost_function(abs(chosen_delta))

        return np.array(list(self.features.values())), self.true_label, (self.group, self.budget, best_modification, max_utility_gain)

    def calculate_utility(self, features, regression_vector):
        x = np.array(list(features.values()))
        logit = np.dot(x, regression_vector)
        probability_not_defaulting = 1 / (1 + np.exp(-logit))
        return probability_not_defaulting

class Learner:
    def __init__(self, true_parameter):
        self.regression_vector = np.array([-0.5, 0.5])  # More realistic initial guess
        self.true_parameter = true_parameter
        self.data = []

    def update_regression_vector(self):
        if not self.data:
            return

        X = np.array([features for features, _ in self.data])
        Y = np.array([outcome for _, outcome in self.data])

        if X.shape[0] > 1:  # Check if there's enough data to perform matrix inversion
            X_transpose = X.T
            self.regression_vector = np.linalg.inv(X_transpose @ X) @ X_transpose @ Y

    def collect_data(self, features, outcome):
        self.data.append((features, outcome))

    def announce_vector(self):
        return self.regression_vector

    def calculate_accuracy(self):
        return np.linalg.norm(self.true_parameter - self.regression_vector) / np.linalg.norm(self.true_parameter)

class Simulation:
    def __init__(self, num_agents, time_steps, true_parameter):
        self.agents = [Agent('advantaged' if i < num_agents // 2 else 'disadvantaged') for i in range(num_agents)]
        self.learner = Learner(true_parameter)
        self.time_steps = time_steps

    def run_simulation(self):
        for t in range(self.time_steps):
            print(f"Time Step {t+1}")
            regression_vector = self.learner.announce_vector()
            for agent in self.agents:
                features, outcome, details = agent.decide_modifications(regression_vector)
                self.learner.collect_data(features, outcome)
                print(f"Agent Details: Group={details[0]}, Remaining Budget={details[1]:.2f}, Feature Modification={details[2]}, Utility Gain={details[3]:.2f}")
            self.learner.update_regression_vector()
            print(f"Current Regression Vector: {self.learner.regression_vector}")
            print(f"Accuracy: {self.learner.calculate_accuracy():.4f}\n")

# Define the true regression parameter
true_parameter = np.array([0.3, -0.2])  # Example true parameter
simulation = Simulation(num_agents=10, time_steps=10, true_parameter=true_parameter)
simulation.run_simulation()
