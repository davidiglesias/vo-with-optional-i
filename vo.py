import os
import sys
import argparse
import numpy as np
import csv
import cv2
from matplotlib import pyplot as plt

from params import ParamsKITTI, ParamsEuroc
from dataset import KITTIOdometry, EuRoCDataset

FIRST_FRAME = 0
SECOND_FRAME = 1
DEFAULT = 2

class VO:
    def __init__(self, path, cam, start_idx=0):
        self.stage = FIRST_FRAME
        self.curr_idx = start_idx
        self.num_processed = 0
        self.max_track_length = 20

        # dataset-dependent params
        self.params = ParamsEuroc()
        self.dataset = EuRoCDataset(path)

        self.detector = cv2.ORB_create(nfeatures=200, scaleFactor=1.2, nlevels=1, edgeThreshold=31)
        self.ffdetector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        self.extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=32, use_orientation=False)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # kpts and descriptors of all frames seen so far
        self.kpts = []
        self.des = []
        self.matches = []

        # params for Shi-Tomasi corner detection
        self.detector_params = dict(maxCorners = 150,
                              qualityLevel = 0.3,
                              minDistance = 7,
                              blockSize = 7)

        # tracker params
        self.tracker_params = dict(winSize = (21, 21),
                                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        # hash table to find 3d points across frames
        #self.pts_3d = {}
        self.pts_3d = []
        self.good_idxs = []
        self.good_trks = []

        # relevant images seen so far
        self.prev_img = self.dataset.left[start_idx]
        self.curr_img = self.dataset.left[start_idx]

        # camera model
        self.f = (cam.fx + cam.fy) / 2 # avg of both focal lengths
        self.pp = (cam.cx, cam.cy)
        self.K = np.append(cam.intrinsic_matrix, np.array([[0, 0, 0]]).T, axis=1) # 3x4 ndarray

        # trajectory
        self.poses = []

        self.viz = True
        self.tracks = []
        self.new_tracks = []
        self.done = False

    def detect(self):
        print("detecting")
        mask = np.zeros_like(self.curr_img)
        mask[:] = 255
        for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
            cv2.circle(mask, (x, y), 5, 0, -1)
        self.new_tracks = []
        p = cv2.goodFeaturesToTrack(self.curr_img, mask = mask, **(self.detector_params))
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                self.new_tracks.append([(x, y)])

    def track(self):
        print("tracking")
        img0, img1 = self.prev_img, self.curr_img
        p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **(self.tracker_params))
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **(self.tracker_params))
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []

        self.good_trks = np.zeros((np.asarray(self.tracks).shape[0],), dtype=np.int8) # should be Nx1
        idx = 0
        for tr, (x, y), is_good in zip(self.tracks, p1.reshape(-1, 2), good):
            #print(self.good_trks)
            if not is_good:
                idx += 1
                continue
            self.good_trks[idx] = 1
            tr.append((x, y))
            if len(tr) > self.max_track_length:
                del tr[0]
            new_tracks.append(tr)
            idx += 1
        self.tracks = new_tracks

    def extract_rel_pose(self, prev_kpts, curr_kpts):
        E, mask = cv2.findEssentialMat(np.array(prev_kpts),
                                       np.array(curr_kpts),
                                       focal=self.f,
                                       pp=self.pp,
                                       method=cv2.RANSAC,
                                       prob=0.99,
                                       threshold=0.5)
        _, R, t, _ = cv2.recoverPose(E,
                                 np.array(prev_kpts),
                                 np.array(curr_kpts),
                                 focal=self.f,
                                 pp=self.pp)
        return R, t

    def solve_pnp(self, kpts):
        print("pts_3d shape:", self.pts_3d.shape[1])
        print("good_trks shape:", len(self.good_trks))
        good_idxs_pnp = self.good_trks[:self.pts_3d.shape[1] + 1].astype(bool)
        print("good_idxs_pnp", good_idxs_pnp, "shape:", len(good_idxs_pnp))
        print("kpts shape:", np.asarray(kpts).shape)
        _, rot, t, inliers = cv2.solvePnPRansac(np.asarray(self.pts_3d[:, good_idxs_pnp]).T,
                                                np.asarray(kpts)[good_idxs_pnp, :], self.K[:,:3], None,
                                                None, None, False, 50, 2.0, 0.9, None)

        R = cv2.Rodrigues(rot)[0]
        return R, t

    def draw(self):
        vis = cv2.cvtColor(self.curr_img, cv2.COLOR_GRAY2BGR)
        for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
            cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)
        cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0), 1)
        cv2.imshow('LK tracker', vis)
        c = cv2.waitKey(0)
        return c

    def update(self):
        print("\n-------------------------------------\nprocessing frame", self.num_processed)
        self.prev_img = self.curr_img
        self.curr_img = self.dataset.left[self.curr_idx]

        if self.stage == FIRST_FRAME:
            R = np.array([[1.0, 0, 0],
                          [0, 1.0, 0],
                          [0, 0, 1.0]]) # rotation matrix
            t = np.array([0, 0, 0]) # translation vector
            self.poses.append((R, t))
            self.stage = SECOND_FRAME

        if self.num_processed % 10 == 0 or len(self.tracks) < 10:
            self.detect()

        print("{} existing tracks, {} new tracks".format(len(self.tracks), len(self.new_tracks)))

        if len(self.tracks) > 0:
            self.track()
            kpts1 = []
            kpts2 = []
            print("{} tracks kept after tracking".format(len(self.tracks)))
            for tr in self.tracks:
                if len(tr) < 2:
                    continue
                kpts1.append(tr[-2])
                kpts2.append(tr[-1])

            if self.stage == SECOND_FRAME:
                R, t = self.extract_rel_pose(kpts1, kpts2)
                self.stage = DEFAULT
                #print(R, t)

            elif self.stage == DEFAULT:
                R, t = self.solve_pnp(kpts2)
                #print(R, t)

            self.pts_3d, self.good_idxs = self.triangulate_points(R, t, kpts1, kpts2)
            print("self.good_idxs.shape:", len(self.good_idxs))
            print("self.good_idxs BEFORE:", len(self.good_idxs))
            self.good_idxs = self.good_idxs[list(self.good_trks)]
            print("self.good_idxs AFTER:", len(self.good_idxs))

        for tr in self.new_tracks:
            self.tracks.append(tr)
        self.new_tracks = []

        if self.viz:
            c = self.draw()
            if c == 27:
                self.done = True

        self.num_processed += 1
        self.curr_idx += 1
        self.prev_img = self.curr_img

    def triangulate_points(self, R, t, kpts1, kpts2):
        P_1 = self.K.dot(np.linalg.inv(self.T_from_Rt(R, t)))
        P_2 = self.K # assume camera 2 is at origin

        pts_hom = cv2.triangulatePoints(P_1, P_2, np.asarray(kpts1).T, np.asarray(kpts2).T) # in homogeneous coords
        pts = pts_hom / np.tile(pts_hom[-1, :], (4, 1)) # 4xN
        good_idxs = (pts[3,:] > 0) & (np.abs(pts[2, :]) > 0.01)
        #print("in triangulate_points... good_idxs:", good_idxs)
        return np.array(pts[:3, good_idxs]), good_idxs # 3xM, where M = len(good_idxs)

    def draw_matches(self):
        plt.ion() # interactive
        if self.stage == DEFAULT:
            img_matches = cv2.drawMatchesKnn(self.prev_image, self.prev_kpts, self.curr_image,
                                             self.curr_kpts, self.matches[-1], None,
                                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.figure(figsize=(12,8), dpi=100)
            plt.imshow(img_matches)
            plt.show()
            input("press any key")

    def T_from_Rt(self, R, t):
        t = t.reshape((3, 1))
        R = R.reshape((3, 3))
        return np.append(np.append(R, t, axis=1), np.array([[0,0,0,1]]), axis=0)

    def get_image(self, idx):
        idx = max(0, idx)
        return self.dataset.left[idx]

    @property
    def curr_image(self):
        return self.curr_img

    @property
    def curr_kpts(self):
        if len(self.kpts) > 0:
            return self.kpts[-1]
        else:
            return []

    @property
    def prev_image(self):
        return self.prev_img

    @property
    def prev_kpts(self):
        if len(self.kpts) > 1:
            print("len(self.kpts):", len(self.kpts))
            return self.kpts[-2]
        else:
            return []

if __name__ == '__main__':
    print("in main()")
    path = '/Users/David/Downloads'
    start_idx = 400
    frames_to_process = 1000
    dataset = EuRoCDataset(path)
    vo = VO(path, dataset.left_cam, start_idx)

    # loop through images
    for i in range(1, 20):
        vo.update()
        if vo.done:
            break

    cv2.destroyAllWindows()
