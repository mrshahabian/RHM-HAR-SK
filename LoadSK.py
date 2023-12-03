import os

import torch
import pickle

class LoadSK():
    def __init__(self, data=None, bins=None, sk_path= 'RH_HAR_skeleton.pkl',
                 cnf_path='RH_HAR_confidence.pkl',
                 bbox_path='RH_HAR_bbox.pkl',
                 labels=None,
                 transpose=False, distance=False):

        if labels is None:
            labels = ['Bending', 'SittingDown', 'ClosingCan', 'Reaching', 'Walking', 'Drinking', 'StairsClimbingUp',
                      'StairsClimbingDown', 'StandingUp', 'OpeningCan', 'CarryingObject', 'Cleaning',
                      'PuttingDownObjects', 'LiftingObject']

        # Get the current directory of the script
        self.current_directory = os.path.dirname(__file__)  # Get the directory of the current Python script
        print(f' self.current_directory: {self.current_directory} ')
        # Navigate to the parent directory
        self.parent_directory = os.path.dirname(self.current_directory)
        print(f' self.parent_directory: {self.parent_directory} ')
        # Define the relative path to your .pkl file
        # in oue case the data is in the same parent directory as the script
        # ( the git repo is besides the data folder (RHM-HAR-SK-Dataset))
        self.pkl_file_paths = os.path.join(self.parent_directory, 'RHM-HAR-SK-Dataset/Yolo7/')

        sk_path = os.path.join(self.pkl_file_paths, sk_path)
        cnf_path = os.path.join(self.pkl_file_paths, cnf_path)
        bbox_path = os.path.join(self.pkl_file_paths, bbox_path)

        # loading the data from the pickle files
        self.skeleton_vectors = self.load_pkl(sk_path)
        self.conf_vectors = self.load_pkl(cnf_path)
        self.bbox_vectors = self.load_pkl(bbox_path)

        # loading the input labels, the defaults values shows 14 actions, but we can change it to any number of actions
        self.labels = labels
        self.number_of_action = len(self.labels)

        # transpose the data if needed
        self.transpose = transpose
        self.distance = distance

    # loading the pickle files method
    def load_pkl(self, path):
        with open(path, 'rb') as f:
            vectors = pickle.load(f)
        return vectors

    def __len_views__(self):
        return len(self.skeleton_vectors)

    def __len_actions__(self, view):
        return len(self.skeleton_vectors[view])

    def __len_samples__(self, view, action):
        return len(self.skeleton_vectors[view][action])

    # this method getting items from loaded skeleton data from .pkl file
    def __getitem__(self, view, action, sample):
        # The transpose is used to change the dimension of the data to be in the form of frames by joints(False), and joints by frames (True)
        if self.transpose:
            return self.skeleton_vectors[view][action][sample].transpose(0, 1)
        else:
            return self.skeleton_vectors[view][action][sample]
    def __getcnf__(self, view, action, sample):
        return self.conf_vectors[view][action][sample]
    def __getbbox__(self, view, action, sample):
        return self.bbox_vectors[view][action][sample]


# data = LoadSK()
# print(f' data: {data.__len_views__()} ')
# print(f' data: {data.__len_actions__(0)} ')
# print(f' data: {data.__len_samples__(0,0)} ')
# print(f' data: {data.__getitem__(0,0,0).shape} ')
# print(f' data: {data.__getcnf__(0,0,0).shape} ')
