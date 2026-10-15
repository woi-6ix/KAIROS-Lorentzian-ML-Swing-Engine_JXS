"""Python numeric mirror of the Pine HMM, NOT a Pine compiler or backtester."""
import math


def emissions(x, means, variances):
    logs = [-0.5 * (math.log(2 * math.pi * v) + (x - m)**2 / v)
            for m, v in zip(means, variances)]
    largest = max(logs)
    return [math.exp(v - largest) for v in logs]


def forward(x, previous, means, variances, transitions):
    emission = emissions(x, means, variances)
    weights = [sum(previous[i] * transitions[i][j] for i in range(3))
               * emission[j] for j in range(3)]
    total = max(sum(weights), 1e-100)
    return [v / total for v in weights]


def expectation(observations, start, means, variances, transitions):
    """Scaled alpha/beta expectation step, following the Pine loop order."""
    n = len(observations)
    alpha, scales, density = [], [], []
    for t, x in enumerate(observations):
        emit = emissions(x, means, variances)
        density.append(emit)
        prior = start if t == 0 else [
            sum(alpha[t-1][i] * transitions[i][j] for i in range(3))
            for j in range(3)]
        weights = [prior[j] * emit[j] for j in range(3)]
        total = max(sum(weights), 1e-100)
        scales.append(total)
        alpha.append([v / total for v in weights])
    occupancy, first, second = [0.0]*3, [0.0]*3, [0.0]*3
    counts, gamma_first = [[0.0]*3 for _ in range(3)], None
    beta = [1.0]*3
    for t in range(n-1, -1, -1):
        gamma = [alpha[t][j] * beta[j] for j in range(3)]
        total = max(sum(gamma), 1e-100)
        gamma = [v/total for v in gamma]
        x = observations[t]
        for j in range(3):
            occupancy[j] += gamma[j]
            first[j] += gamma[j]*x
            second[j] += gamma[j]*x*x
        if t == 0:
            gamma_first = gamma
        if t > 0:
            previous_beta = [0.0]*3
            xi = [[0.0]*3 for _ in range(3)]
            for i in range(3):
                for j in range(3):
                    link = transitions[i][j]*density[t][j]*beta[j]
                    previous_beta[i] += link/scales[t]
                    xi[i][j] = alpha[t-1][i]*link
            total = max(sum(map(sum, xi)), 1e-100)
            for i in range(3):
                for j in range(3):
                    counts[i][j] += xi[i][j]/total
            beta = previous_beta
    return occupancy, first, second, counts, gamma_first


def fit(observations, iterations=5, floor=0.01):
    n = len(observations)
    mean = sum(observations)/n
    variance = max(sum((x-mean)**2 for x in observations)/n, floor)
    means = [mean+(j-1)*math.sqrt(variance) for j in range(3)]
    variances = [variance]*3
    transitions = [[0.95 if i == j else 0.025 for j in range(3)] for i in range(3)]
    start = [1/3]*3
    for _ in range(iterations):
        occ, first, second, counts, start = expectation(
            observations, start, means, variances, transitions)
        for j in range(3):
            if occ[j] > 1e-6:
                means[j] = first[j]/occ[j]
                variances[j] = max(second[j]/occ[j]-means[j]**2, floor)
            weights = [v+0.01 for v in counts[j]]
            transitions[j] = [v/sum(weights) for v in weights]
    order = sorted(range(3), key=means.__getitem__)
    means = [means[i] for i in order]
    variances = [variances[i] for i in order]
    transitions = [[transitions[i][j] for j in order] for i in order]
    start = [start[i] for i in order]
    posterior = None
    for t, x in enumerate(observations):
        if t == 0:
            e = emissions(x, means, variances)
            weights = [start[j]*e[j] for j in range(3)]
            posterior = [v/max(sum(weights), 1e-100) for v in weights]
        else:
            posterior = forward(x, posterior, means, variances, transitions)
    return means, variances, transitions, posterior


def gate(side, enabled, ready, state, run, confidence, means,
         minimum_confidence=0.55, confirmation=2, drift=0.05):
    if not enabled:
        return True
    if not ready or run < confirmation or confidence < minimum_confidence:
        return False
    return ((side == 1 and state == 2 and means[2] > drift)
            or (side == -1 and state == 0 and means[0] < -drift))


class Stream:
    def __init__(self, window=100, interval=25, iterations=3):
        self.window, self.interval, self.iterations = window, interval, iterations
        self.history, self.fit_windows = [], []
        self.observations = 0
        self.last_fit, self.last_bar = None, None
        self.state, self.run = 1, 0
        self.means = self.variances = self.transitions = self.posterior = None
        self.position, self.closed = 0, 0

    def execution(self, bar, x, *, enabled=True, confirmed=True, position=0, closed=0):
        fill = position != self.position or closed != self.closed
        self.position, self.closed = position, closed
        if not enabled or not confirmed or fill or self.last_bar == bar:
            return None
        self.last_bar = bar
        if x is None:
            self.run = 0
            return None
        refit = self.last_fit is None or self.observations-self.last_fit >= self.interval
        if len(self.history) == self.window and refit:
            self.fit_windows.append((bar, tuple(self.history)))
            self.means, self.variances, self.transitions, self.posterior = fit(
                self.history, self.iterations)
            self.last_fit = self.observations
            self.run = 0
        result = None
        if self.posterior is not None:
            self.posterior = forward(x, self.posterior, self.means, self.variances, self.transitions)
            state = max(range(3), key=self.posterior.__getitem__)
            self.run = self.run+1 if state == self.state else 1
            self.state = state
            result = (state, self.run, tuple(self.posterior))
        self.history = (self.history+[x])[-self.window:]
        self.observations += 1
        return result
