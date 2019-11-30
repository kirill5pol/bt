import numpy as np


def avg_skill_fn():
    """Return the average skill level of a random student."""
    return np.random.randn()


def specific_skill_fn(avg_skill):
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
    return c + (1 - c) / (1 + np.exp(a * (theta - b)))


def n_concept_irt(thetas, a=1, b=0, c=0.25):
    """Item Response Theory with n concepts.

        thetas: skill levels of student for given concepts
        a: slope of signmoid (how sharp the cutoff is)
        b: question difficulty
        c: probability of random guess being correct

    Take the lowest skill level for each of the concepts and use that as the 
    student's skill. Intuition: the weakest skill will be the thing keeping the
    from answering incorrectly.

    Return:
        prob_correct: probability of student getting the answer correct
    """

    ### TODO: Research whether there should be only one difficulty, vs multiple
    #         for this question.
    print("len(theta)", len(theta))
    if len(theta) == 1:
        return one_concept_irt(theta=thetas[0], a=a, b=b, c=c)
    else:
        weakest_skill = thetas[0]
        for theta in thetas[1:]:
            if theta < weakest_skill:
                weakest_skill = theta

        return one_concept_irt(theta=weakest_skill, a=a, b=b, c=c)


def exponential_forgetting_curve(x):
    raise NotImplementedError


def half_life_regression(x):
    raise NotImplementedError


def generalized_power_law(x):
    raise NotImplementedError


def weiner_process(x):
    raise NotImplementedError
