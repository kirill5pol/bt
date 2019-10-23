import numpy as np

from generate import learner_style, learner_skills
from statistics import avg_skill_fn, specific_skill_fn, one_concept_irt, n_concept_irt


class Student(object):
    def __init__(self, n_concepts, avg_skill_fn=np.random.randn):
        self.learner_style = learner_style()
        self.avg_skill = avg_skill_fn()
        self.n_concepts = n_concepts

        self.skills = learner_skills(n_concepts, self.avg_skill, specific_skill_fn)

    def question(question, a=1, c=0.25):
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
            n_concept_irt(concept_skill, b=difficulty)
        elif len(concepts) > 1:
            concept_idxs = concepts
            concept_skills = self.skills[concept_idxs]
            n_concept_irt(concept_skills, b=difficulty)
        else:
            raise ValueError(
                "There must be at least one concept associated with the question {}".format(
                    concepts
                )
            )

        # Sample from p_correct, return True/False
        return (np.random.rand() > p_correct, p_correct)

    def forget(forgetting_type=None):
        """Exponential forgetting function
        
        Two papers with forgetting:
            A) http://people.eecs.berkeley.edu/~reddy/files/DRL_Tutor_NIPS17_MT_Workshop.pdf
            B) https://arxiv.org/pdf/1604.02336.pdf

        A) Each skill exponentially decays according to a Wiener process
        B) Exponential forgetting curve, Half-life regression, & Generalized power-law
        """
        if forgetting_type == "exponential_forgetting_curve":
            self.skills = exponential_forgetting_curve(self.skills, **kwargs)
        elif forgetting_type == "half_life_regression":
            self.skills = half_life_regression(self.skills, **kwargs)
        elif forgetting_type == "generalized_power_law":
            self.skills = generalized_power_law(self.skills, **kwargs)
        elif forgetting_type == "wiener_process":
            self.skills = wiener_process(self.skills, **kwargs)
        else:
            raise ValueError(
                "forgetting_type must be one of the following: {}".format(
                    [
                        exponential_forgetting_curve,
                        half_life_regression,
                        generalized_power_law,
                        wiener_process,
                    ]
                )
            )

    def example(example, delta_scale=1):
        """Showing an example will increase the skill of the student on that 
        particular concept. The increase is proportional to the dot product 
        between the example learning style and student learning style.
             
            example: a tuple of concept index and learning style index
        """
        concept_idx, ls_idx = example
        self.skills[concept_idx] += delta_scale * self.learner_style[ls_idx]

        ### OLD:
        # """
        #      example: a tuple of concept index and learning style of the example
        # """
        # concept_idx, learning_style = example
        # delta_skill = learning_style.dot(self.learner_style)
        # self.skills[concept_idx] += delta_skill

        ### ALSO OLD:
        # """
        #      example: a c by 4 matrix, where c is the number of concepts, and 4
        #         is the number of learning styles. All rows except for one should
        #         be all zeros. Only a single active concept row should be
        #         non-zero. Each column represents one of the 4 skills.
        # """
        # # Shapes: (c,) = (c,4) * (4,)
        # delta_skills = example.dot(self.learner_style)
        # self.skills[:] += delta_skills
