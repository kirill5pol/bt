import numpy as np


class Agent(object):
    """Agent base class"""

    def __init__(self, n_concepts, **kwargs):
        super(Agent, self).__init__(**kwargs)
        self.n_concepts = n_concepts
        self.n_learning_styles = 4

    def __call__(self, state, reward, done, info):
        raise NotImplementedError

    def save(self, seed=None):
        if seed == None:
            seed = np.random.randint(1000)
        filename = f"data/{seed}-{type(self).__name__}"
        with open(filename, "wb") as f:
            pickle.dump(self.__dict__, f)
        print("Saved neural network parameters to:", filename)

    def load(self, seed):
        filename = f"data/{seed}-{type(self).__name__}"
        with open(filename, "rb") as f:
            self.__dict__ = pickle.load(f)


class RandomAgent(Agent):
    """Picks a random skill and learning style to teach the student."""

    def __call__(self, state, reward, done, info):
        concept = np.random.choice(self.n_concepts)
        l_style_category = np.random.choice(self.n_learning_styles)
        return [concept * self.n_learning_styles + l_style_category]


class WeakestSkillAgent(Agent):
    """Looks at each student's skill vector and learning style vector and uses 
    that information to decide what to teach next. Picks weakest skill and most
    dominant learner style.
    """

    def __call__(self, state, reward, done, info):
        student_skills = info[0]["student_skills"]
        weakest_skill = np.argmin(student_skills)
        concept = weakest_skill

        student_learner_style = info[0]["student_learner_style"]
        strongest_l_style = np.argmax(student_learner_style)
        l_style_category = strongest_l_style

        return [concept * self.n_learning_styles + l_style_category]


class MultiArmBandit(Agent):
    """Implement a MultiArmBandit for learning. Uses a Monte Carlo evaluation to
    determine the expected reward from an action.
    Note: this is only the discrete version.

    Expected reward function (expected reward for each action): Q_t(a)
    We assume after enough Monte Carlo samples we can estimate Q(a) by:
    Q̂_t(a) ≈ Q(a)
    By using the real rewards from monte carlo:
    Q̂_t(a) = Q̂_{t-1}(a) + (1 / N_t(a)) (r_t - Q̂_{t-1})
    """

    def __init__(self, n_concepts, eps=lambda n: 0.1, **kwargs):
        """
        Inputs:
            n_concepts: Number of concepts in problem
            eps: Epsilon in the epsilon-greedy algorithm. Is a function that 
                takes current number of steps (allows for decaying epsilon)
        """
        super(MultiArmBandit, self).__init__(n_concepts, **kwargs)
        self.n = 1  # Total number of steps taken
        self.eps = eps

        action_shape = (self.n_concepts * self.n_learning_styles,)
        self.q = np.zeros(action_shape)  # Estimated expected reward for each action

        self.prev_action = 0


class MultiArmBanditEpsilonGreedy(MultiArmBandit):
    def __call__(self, state, reward, done, info):
        if info[0]["question_idx"] == 0:  # Means we just started teaching a new student
            self.prev_action = 0

        # Expected reward update
        #  Q̂_t(a) = Q̂_{t-1}(a) + (1 / N_t(a)) (r_t - Q̂_{t-1})
        pa = self.prev_action
        self.q[pa] = self.q[pa] + (1 / self.n) * (reward - self.q[pa])

        if self.n < 5 * self.q.shape[0]:  # Take random action for estimating
            # Random action
            action = np.random.choice(self.q.shape[0])
        elif np.random.rand() > self.eps(self.n):  # Epsilon-greedy portion
            # Greedy action (take the highest expected reward action)
            action = np.argmax(self.q)
        else:
            # Random action
            action = np.random.choice(self.q.shape[0])

        self.prev_action = action  # For the next expected reward update
        return [action]


class MultiArmBanditEpsilonSampleProb(MultiArmBandit):
    """Difference Between Eps greedy is that for the 'greedy' portion takes an 
    action with a probability corresponding to the softmax of the Q̂ values.
    **Should work better than the Eps-greedy bandit due to the problem actually
    having a state...** 
    """

    def __call__(self, state, reward, done, info):
        if info[0]["question_idx"] == 0:  # Means we just started teaching a new student
            self.prev_action = 0

        # Expected reward update
        #  Q̂_t(a) = Q̂_{t-1}(a) + (1 / N_t(a)) (r_t - Q̂_{t-1})
        pa = self.prev_action
        self.q[pa] = self.q[pa] + (1 / self.n) * (reward - self.q[pa])

        if self.n < 5 * self.q.shape[0]:  # Take random action for estimating
            # Random action
            action = np.random.choice(self.q.shape[0])
        elif np.random.rand() > self.eps(self.n):  # Epsilon-greedy portion
            # Greedy action (take the highest expected reward action)
            action = np.argmax(self.q)
        else:
            # Random action
            # Get probability of each action (softmax of Q̂)
            if np.sum(self.q) == 0:  # First step ever taken
                action = 0
            else:
                prob_actions = self.q / (np.sum(self.q))
                action = np.random.choice(self.q.shape[0], p=prob_actions)

        self.prev_action = action  # For the next expected reward update
        self.n += 1
        return [action]
