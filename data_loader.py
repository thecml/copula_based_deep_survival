import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from typing import List

def get_trajectory_labels(labs):
    unique_labs = np.unique(labs, axis=0)
    new_labs = np.zeros((labs.shape[0],))
    for i in range(labs.shape[0]):
        for j in range(unique_labs.shape[0]):
            if np.all(unique_labs[j, :] == labs[i, :]):
                new_labs[i] = j
    return new_labs

class RotterdamDataLoader():
    """
    Data loader for Rotterdam dataset
    """
    def load_data(self, n_samples:int = None):
        df = pd.read_csv('data/rotterdam.csv')
        if n_samples:
            df = df.sample(n=n_samples, random_state=0)
        self.X = df.drop(['pid', 'rtime', 'recur', 'dtime', 'death'], axis=1)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        times = [df['rtime'].values, df['dtime'].values]
        events = [df['recur'].values, df['death'].values]
        self.y_t = np.stack((times[0], times[1]), axis=1)
        self.y_e = np.stack((events[0], events[1]), axis=1)
        self.n_events = 2
        return self
    
    def split_data(self,
                   train_size: float,
                   valid_size: float,
                   random_state=0):
        # Split multi event data
        raw_data = self.X
        event_time = self.y_t
        labs = self.y_e
        
        traj_labs = labs
        if labs.shape[1] > 1: 
            traj_labs = get_trajectory_labels(labs)

        #split into training/test
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size,
                                          random_state=random_state)
        train_i, test_i = next(splitter.split(raw_data, traj_labs))

        train_data = raw_data.iloc[train_i, :]
        train_labs = labs[train_i, :]
        train_event_time = event_time[train_i, :]

        pretest_data = raw_data.iloc[test_i, :]
        pretest_labs = labs[test_i, :]
        pretest_event_time = event_time[test_i, :]

        #further split test set into test/validation
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=valid_size)
        new_pretest_labs = get_trajectory_labels(pretest_labs)
        test_i, val_i = next(splitter.split(pretest_data, new_pretest_labs))
        test_data = pretest_data.iloc[test_i, :]
        test_labs = pretest_labs[test_i, :]
        test_event_time = pretest_event_time[test_i, :]

        val_data = pretest_data.iloc[val_i, :]
        val_labs = pretest_labs[val_i, :]
        val_event_time = pretest_event_time[val_i, :]

        #package for convenience
        train_pkg = [train_data, train_event_time, train_labs]
        valid_pkg = [val_data, val_event_time, val_labs]
        test_pkg = [test_data, test_event_time, test_labs]

        return (train_pkg, valid_pkg, test_pkg)
    
    def get_data(self) -> pd.DataFrame:
        """
        This method returns the features and targets
        :returns: X, y_t and y_e
        """
        return self.X, self.y_t, self.y_e

    def get_features(self) -> List[str]:
        """
        This method returns the names of numerical and categorial features
        :return: the columns of X as a list
        """
        return self.num_features, self.cat_features

    def _get_num_features(self, data) -> List[str]:
        return data.select_dtypes(include=np.number).columns.tolist()

    def _get_cat_features(self, data) -> List[str]:
        return data.select_dtypes(['object']).columns.tolist()