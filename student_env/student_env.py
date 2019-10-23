import numpy as np
import argparse
import generate
import gym
import os

from student_simulator import Student
from gym.utils import seeding
from copy import deepcopy
from gym import spaces


class StudentEnv(gym.Env):
    def __init__(
        self, load=True, save=None, n_students=20, n_concepts=5, n_questions=100, seed=9
    ):
        """Create the student environment

            load: either None, True, or a filename. If True: use the other 
                information given to the constuctor to guess the filename
            save: location to save. If None: use the other information given to
                the constuctor to create the filename
            n_students: # of students
            n_concepts: # of concepts
            n_questions: # of questions
            seed: global seed
        """
        np.random.seed(seed)

        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(4 * n_concepts)

        # Set the current students to the initial state
        self.students = deepcopy(self.students_init)

        # Create the questions
        self.questions = generate.questions(
            n_questions, n_concepts, max_concepts=1, difficulty_fn=np.random.randn
        )

        # Handle loading and saving to/from an initial state
        self.load_and_save(
            load=load,
            save=save,
            n_students=n_students,
            n_concepts=n_concepts,
            n_questions=n_questions,
            seed=seed,
        )

        self.n_students = n_students
        self.n_concepts = n_concepts
        self.n_questions = n_questions
        self.i = 0  # Current step
        self.s = 0  # Current student
        self.q = 0  # Current question

    def load_and_save(self, load, save, n_students, n_concepts, n_questions, seed):
        if load:
            # Use default load location if load True but not a filename
            if load != True:
                load = f"/data/🧠{n_concepts}_👨‍🎓{n_students}_❓{n_questions}_🌱{seed}"
            self.students_init, self.questions, n_concepts, n_students, n_questions, seed = pkl.load(
                load
            )
        else:
            self.students_init = [Student(n_concepts) for i in range(n_students)]

        # Use default save location if save is none
        save = save or f"/data/🧠{n_concepts}_👨‍🎓{n_students}_❓{n_questions}_🌱{seed}"
        o = (
            self.students_init,
            self.questions,
            n_concepts,
            n_students,
            n_questions,
            seed,
        )
        pkl.dump(save, o)

    def reset(self):
        self.students = deepcopy(self.students_init)
        return self.step(np.zeros(self.action_space.shape))  # Do no action

    def step(self, action):
        student = self.students[self.s]
        question = self.questions[self.q]
        concept_idx = int(action / n_concepts)
        learning_style_idx = action % n_concepts
        example = (concept_idx, learning_style_idx)

        # Show the student the next example
        student.example(example)

        # Ask the student the next question
        correct, p_correct = student.question(question)
        reward = int(correct)  # Reward of 1 if correct answer

        # Increment steps, the question, and the possibly the student
        self.i += 1
        self.q = (self.q + 1) % self.n_questions
        if i % self.n_questions == 0:  # Next student once student completes all qs
            self.s = (self.s + 1) % self.n_students

        # Done episode if all students have been shown all questions
        done = self.i >= self.n_students * self.n_questions
        info = {}

        # What is the state? -> The knowledge space of each student
        # What should the obersvation be? Whether the student answered the
        #    current question correctly or not (so same as reward I guess??).
        #    To use the obs, you need to use an RNN.
        state = int(correct)

        return state, reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass
