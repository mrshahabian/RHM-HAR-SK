import pickle

class SamplingRHy7():
    def __init__(self,data=None, bins=None, sk_path= 'RH_HAR_skeleton.pkl' , cnf_path= 'RH_HAR_confidence.pkl', bbox_path= 'RH_HAR_bbox.pkl',
                 labels= ['Bending', 'SittingDown', 'ClosingCan', 'Reaching', 'Walking', 'Drinking', 'StairsClimbingUp',
                          'StairsClimbingDown', 'StandingUp', 'OpeningCan', 'CarryingObject', 'Cleaning', 'PuttingDownObjects', 'LiftingObject'],
                 transpose=False, distance=False):
        super().__init__(data,bins)

        # loading the data from the pickle files
        self.skeleton_vectors = self.load_pkl(sk_path)
        self.conf_vectors = self.load_pkl(cnf_path)
        self.bbox_vectors = self.load_pkl(bbox_path)

        # loading the input labels, the defults values shows 14 actions, but we can change it to any number of actions
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
            return self.skeleton_vectors[view][action][sample].transpose(0,1)
        else:
            return self.skeleton_vectors[view][action][sample]