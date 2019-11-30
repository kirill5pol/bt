from collections import namedtuple
import numpy as np


Question = namedtuple("Question", "concepts difficulty")


def concepts(n):
    """Create `n` concepts where each is named with a letter or series of letters
    ie A, B, ..., Z, AA, AB, ...

    Return concepts: a dictionary that maps the concept number to it's name
    ie. {0:'A', ..., 25:'Z', 26:'AA', 27:'AB', ...}
    """

    def letters(n):
        if n < 26:
            return chr(ord("A") + n)  # Get the letter
        else:
            return letters(int(n / 26) - 1) + chr(ord("A") + n % 26)  # Get the letters

    concepts = {}
    concepts_inv = {}
    i = 0
    while i < n:
        concept_name = letters(i)
        concepts[i] = concept_name
        concepts_inv[concept_name] = i
        i += 1

    return concepts, concepts_inv


def q_difficulty_function(concepts):
    concept = concepts[0]
    skill_difficulty = {
        0: 10,  # Difficult
        1: 3,  # Moderate
        2: 0,  # Moderate
        3: -4,  # Easy
        4: 20,  # Extremely difficult
    }
    if concept < 5:
        return skill_difficulty[concept] + np.random.randn()
    else:
        return np.random.randn()


def questions(q, c, max_concepts=3, difficulty_fn=q_difficulty_function):
    """Generate q questions that sample from the c concepts

        q: nmuber of questions to generate
        c: number of concepts to sample from
        max_concepts: the maximum number of concepts each question can contain
        difficulty_fn: returns a value that represents the difficulty (is an 
            input into an IRT function)
    
    Return qs: A list of namedtuples with (concepts, difficulty):
        where concepts is a tuple of concepts that a question contains, and 
        difficulty is a gaussian 
    """

    def norm_inv_power_law(n):
        p = np.array([1 / i for i in range(1, n + 1)])  # Unnormalized inv power law
        return p / np.sum(p)  # Normalize so sum of probabilities == 1

    qs = []
    for i in range(q):
        # Have between 1 and `max_concepts` concepts, where the probability of
        # number of concepts match the normailized inverse power law
        n_concepts = np.random.choice(max_concepts, p=norm_inv_power_law(max_concepts))

        # Get which concepts this question should be (sample w/out replacement)
        concepts = tuple(np.random.choice(c, size=n_concepts + 1, replace=False))
        # difficulty = difficulty_fn(concepts[0])
        difficulty = q_difficulty_function(concepts)

        qs.append(Question(concepts, difficulty))

    return qs


def learner_style():

    """Create a `learner_style` representation for a student using `VARK`

    Return learner_style: a vector that represents the learning style.
        Categories of learner (in order of index of output) are:
            0: visual (V)
            1: aural (A)
            2: reading/writing (R)
            3: kinesthetic (K)

    ----------------------------------------------------------------------------
    Methods:
    The paper `Attempted Validation of the Scores of the VARK` was able to 
    approximately match learning styles to students by using the correlated 
    trait–correlated uniqueness (CTCU) model. 

    Papers: 
        - Mz14CoendersSaris.pdf
        - Attempted Validation of the Scores of the VARK.pdf
    """

    if True:
        # One learner style
        ls = np.zeros(4)
        ls[np.random.choice(4)] = 1.0
        return ls
    else:
        # Version 1: TODO fancier sampling of the learner styles
        # From the sample in the paper. We take this to be a reasonable starting point
        one_ls = 0.297  # Number of people with only one learning style
        four_ls = 0.358  # Number of people with all learning styles
        # Assume 2 and 3 learning styles are approximately the equal
        two_ls = (1 - one_ls - four_ls) / 2
        three_ls = (1 - one_ls - four_ls) / 2

        rand_num = np.random.rand()
        if rand_num < one_ls:
            # One learner style
            ls = np.zeros(4)
            ls[np.random.choice(4)] = 1.0
            return ls

        elif rand_num < one_ls + two_ls:
            # Two learner styles
            ls = np.clip(np.random.randn(4) + 1, a_min=0.0, a_max=None) + 0.001
            ls[
                np.random.choice(4, size=2, replace=False)
            ] = 0  # randomly set 2 values to 0
            return ls / np.sum(ls)

        elif rand_num < one_ls + two_ls + three_ls:
            # Three learner styles
            ls = np.clip(np.random.randn(4) + 1, a_min=0.0, a_max=None) + 0.001
            ls[np.random.choice(4)] = 0  # randomly set 1 values to 0
            return ls / np.sum(ls)

        elif rand_num < one_ls + two_ls + three_ls + four_ls:
            # All learner styles
            ls = np.clip(np.random.randn(4) + 1, a_min=0.0, a_max=None) + 0.001
            return ls / np.sum(ls)

        else:
            # should never happen
            raise ValueError(
                "Random number was above 1: rand_num:{}, 1-4_ls:{}".format(
                    rand_num, (one_ls, two_ls, three_ls, four_ls)
                )
            )


def learner_skills(c, avg_skill=0, specific_skill_fn=None):
    """Create c learner skills using specific_skill_fn and avg_skill"""
    skills = np.zeros(c)
    for i in range(c):
        skills[i] = specific_skill_fn(avg_skill, i)
    return skills
