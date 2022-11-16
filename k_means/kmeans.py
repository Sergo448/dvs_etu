# project name: dvs_etu
# version: 1.0
# file name: kmeans.py
# auther: Sergo448, 7193
# date: 15.11.2021
# Python version : 3.10

from Funks import Make_funks
import random
import cv2
import numpy as np


class K_means:
    def __init__(self, img_path, num_clussters):
        self.v_sets = []
        self.num_clusters = num_clussters
        # self.img = None
        self.img = cv2.imread(img_path)
        self.img_clustered = None

    # Initialize Codebook that include k codevectors
    def codebook_initialization(self, ):
        tmp = []
        (rows, cols, _) = self.img.shape

        self.img_clustered = [[0 for i in range(rows)] for j in range(cols)]

        # chose k random points as initial clusters (without replacement)
        for i in range(0, self.num_clusters):

            repeat = False
            while True:
                row = random.randint(0, rows)
                col = random.randint(0, cols)

                # assure without replacement
                for codebook in tmp:
                    if row and col in codebook:
                        repeat = True
                        break

                if not repeat:
                    break

            codevector = [self.img[row][col][0], self.img[row][col][1], self.img[row][col][2]]
            v_set = Make_funks(codevector, i)
            self.v_sets.append(v_set)
            tmp.append([row, col])

    @staticmethod
    def euclidean_distance(p1, p2):
        return pow(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2) + pow(p1[2] - p2[2], 2), 1 / 2)

    # For each point, find the nearest codevector
    def e_step(self, ):
        for i in range(len(self.img_clustered)):
            for j in range(len(self.img_clustered[i])):

                point = [self.img[j][i][0], self.img[j][i][1], self.img[j][i][2]]

                dist = float("inf")
                v_set_track = None
                for v_set in self.v_sets:
                    new_dist = self.euclidean_distance(point, v_set.get_codevector())
                    if new_dist < dist:
                        dist = new_dist
                        v_set_track = v_set

                self.img_clustered[i][j] = v_set_track.get_ID()
                v_set_track.add_sum(point)

    # For each codevector, compute the new position (mean).
    # When no codevector changes position, we have come to convergence.
    def m_step(self):
        convergence = True
        for v_set in self.v_sets:
            new_codevector = v_set.get_new_codevector()
            current_codevector = v_set.get_codevector()
            print("current codevector : ", current_codevector)
            print("new codevector : ", new_codevector)
            if new_codevector != current_codevector:
                convergence = False
                v_set.set_codevector(new_codevector)

        return convergence

    # Color for each pixel is equal to the corrispondent codevector.
    def image_clustered(self):
        img = self.img.copy()

        for i in range(len(self.img_clustered)):
            for j in range(len(self.img_clustered[i])):
                for v_set in self.v_sets:
                    if self.img_clustered[i][j] == v_set.get_ID():
                        img[j][i] = v_set.get_codevector()
                        break
        return img

    def clustering(self, img, num_clusters):

        self.num_clusters = num_clusters
        self.img = img
        self.codebook_initialization()

        steps = 1
        while True:
            print("Step number : ", steps)
            self.e_step()
            convergence = self.m_step()
            for v_set in self.v_sets: v_set.reset_sum()
            if convergence:
                break
            steps = steps + 1

        return self.image_clustered()
