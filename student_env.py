from student_simulator import Student
from gym.utils import seeding
from copy import deepcopy
from gym import spaces
import pickle as pkl
import numpy as np
import argparse
import generate
import gym
import os


def one_hot(n, shape):
    x = np.zeros(shape, dtype=np.int32)
    x[n] = 1
    return x


class StudentEnv(gym.Env):
    def __init__(
        self,
        load=False,
        save=None,
        n_students=20,
        n_concepts=5,
        n_questions=500,
        seed=9,
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

        self.observation_space = spaces.MultiDiscrete([n_students, n_questions])
        self.action_space = spaces.Discrete(4 * n_concepts)

        self.n_students = n_students
        self.n_concepts = n_concepts
        self.n_lstyles = 4  # VARK learning styles
        self.n_questions = n_questions
        self.i = 0  # Current step
        self.s = 0  # Current student
        self.q = 0  # Current question
        self.max_steps = self.n_students * self.n_questions

        # Handle loading and saving to/from an initial state
        self.load(filename=load)

        # Set the current students to the initial state
        self.students = deepcopy(self.students_init)

        # Create the questions
        self.questions = generate.questions(
            n_questions, n_concepts, max_concepts=1, difficulty_fn=np.random.randn
        )

    def load(self, filename, seed=None):
        if seed is None:
            seed = np.random.randint(1000)
        if filename:
            # Use default filename location if filename True but not a filename
            if filename != True:
                filename = f"/data/🧠{self.n_concepts}_👨‍🎓{self.n_students}_❓{self.n_questions}_🌱{seed}"
            self.students_init, self.questions, self.n_concepts, self.n_students, self.n_questions, seed = pkl.load(
                filename
            )
        else:
            self.students_init = [
                Student(self.n_concepts) for i in range(self.n_students)
            ]

    def save(self, filename, seed=None):
        if seed is None:
            seed = np.random.randint(1000)
        # Use default filename location if filename is none
        if filename is None:
            filename = f"/data/🧠{self.n_concepts}_👨‍🎓{self.n_students}_❓{self.n_questions}_🌱{seed}"
        o = (
            self.students_init,
            self.questions,
            self.n_concepts,
            self.n_students,
            self.n_questions,
            seed,
        )
        pkl.dump(filename, o)
        print(f"Saving to {filename}")

    def reset(self, shuffle_students=False):
        self.i = 0  # Current step
        self.s = 0  # Current student
        self.q = 0  # Current question
        self.students = deepcopy(self.students_init)
        if shuffle_students:  # Does it make sense to ever shuffle the students?
            self.students = list(np.random.shuffle(self.students))
        return 0  # Default state

    def step(self, action):
        if isinstance(action, (np.ndarray, list)):
            action = action[0]
        student = self.students[self.s]
        question = self.questions[self.q]
        concept_idx = int(action / self.n_lstyles)
        learning_style_idx = action % self.n_lstyles
        ex = (concept_idx, learning_style_idx)

        # Show the student the next example
        student.example(ex)

        # Ask the student the next question
        correct, p_correct = student.question(question)
        reward = int(correct)  # Reward of 1 if correct answer

        # Increment steps, the question, and the possibly the student
        self.i += 1
        self.q = (self.q + 1) % self.n_questions
        if self.i % self.n_questions == 0:  # Next student once student completes all qs
            self.s = (self.s + 1) % self.n_students

        # Done episode if all students have been shown all questions
        done = self.i >= self.max_steps - 1

        # Give the true knowledge state of the student
        info = {
            "student_idx": self.s,
            "question_idx": self.q,
            "student_skills": student.skills,
            "student_learner_style": student.learner_style,
        }

        # What is the state? -> The knowledge space of each student
        # What should the obersvation be? Whether the student answered the
        #    current question correctly or not (so same as reward I guess??).
        #    To use the obs, you need to use an RNN.

        # State is the student and question being asked
        state = np.array([self.s, self.q])

        return state, reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass
