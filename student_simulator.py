import numpy as np

from generate import learner_style, learner_skills


# ==============================================================================
# For some reason these don't want to import.......... >:(
def avg_skill_fn():
    """Return the average skill level of a random student."""
    return np.random.randn()


def specific_skill_fn(avg_skill, skill_idx):
    """Return the skill level of a student on a particular concept, given their
    average skill level.
    """
    # # Some chance that the student never learned the skill, (skill = 0)
    # p_new_skill = 0.75
    # if np.random.rand() < p_new_skill:
    #     return 0.0
    return np.random.randn() + avg_skill


def one_concept_irt(theta, a=1, b=0, c=0.25):
    """Item Response Theory with one concept.

        theta: skill level of student (concept_skill)
        a: slope of signmoid (how sharp the cutoff is)
        b: question difficulty
        c: probability of random guess being correct

    Return:
        prob_correct: probability of student getting the answer correct
    """
    return c + (1 - c) / (1 + np.exp(a * (b - theta)))


irt = one_concept_irt

# ==========================================================================


class Student(object):
    def __init__(self, n_concepts, avg_skill_fn=lambda: np.random.randn() - 3.0):
        """
        avg_skill_fn: 
        If you have: avg_skill_fn=lambda: np.random.randn() - 3.0
        Then you get average skills like:
            [-2.178, -3.758, -3.262, -2.667, -3.622, -2.654, -5.108, -3.844, -3.450, -2.402]

        If your students avg_skill_fn == -2.178:
        The specfic skill will be something like:
            [-2.877, -2.840, -2.464, -2.606, -1.544]

        With a questions of each concept of difficulty 0, the student will get 
        prob correct answer like:
            [0.289, 0.291, 0.308, 0.301, 0.381]

        """
        self.learner_style = learner_style()
        self.avg_skill = avg_skill_fn()
        self.n_concepts = n_concepts

        self.skills = learner_skills(n_concepts, self.avg_skill, specific_skill_fn)

    def question(self, question, a=1, c=0.25):
        """Take a question and return the probability that the student answers
        correctly.

            question: a tuple of concepts in the question, and the question
                difficulty
            a: the slope of the sigmoid when at (1-c)/2
            c: probability of a random guess being correct (lower bound to the 
                probability correct)

        For one concept:
            The probability of a student getting a exercise with difficulty b
            correct if the student had concept skill θ is modelled using
            classic Item Response Theory [DH90] as:
            p(correct|θ, a, b, c) = c + (1 - c) / (1 + exp(a(θ - b))
            where a is the scaling factor of the sigmoid and c is the 
            probability of a random guess (set to be 0.25).

        For multiple concepts per question:
            Use the weakest skill as the maximum skill level for the question

        Returns tuple (answer_correct, p_correct)
            p_correct: probability of getting the question correct
            answer_correct: True/False, whether a student answered the
                question correctly or not.
        """
        concepts, difficulty = question

        if len(concepts) == 1:
            concept_idx = concepts[0]
            concept_skill = self.skills[concept_idx]
            p_correct = one_concept_irt(concept_skill, b=difficulty)
        else:
            raise ValueError("There must be only one concept for each question")

        # Sample from p_correct, return True/False
        return (np.random.rand() < p_correct, p_correct)

    def example(self, example, delta_scale=0.2):
        """Showing an example will increase the skill of the student on that 
        particular concept. The increase is proportional to the dot product 
        between the example learning style and student learning style.
             
            example: a tuple of concept index and learning style index
        """
        concept_idx, ls_idx = example
        self.skills[concept_idx] += delta_scale * self.learner_style[ls_idx]
