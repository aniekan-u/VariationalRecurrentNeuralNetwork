import os
import numpy as np
import pandas as pd
import torch.utils.data as data
from nlb_tools.nwb_interface import NWBDataset



class NWB(data.Dataset):
    def __init__(self, experiment, train, resample_val, seq_len, neur_count, shuffle=False, seq_start_mode='all', transform=None):

        '''
        INPUT
        experiment     int from 1-4 chooses experiment (1: MC_Maze, 2: MC_RTT, 3: Area2_Bump, 4: DMFC_RSG)
        train          bool true -> train, false -> test
        resample_val   int determines factor of resampling,    1 returns original, once resampled need to redownload to get original (delete folder)
        seq_len        int length of individual sequences
        neur_count     int count of neurons per trial,         0 for full length
        seq_start_mode str                                     ['all', 'unique']
        tranform       function                                

        OUTPUT
        neuron_id      2d int-array of neuron ids in data
        trial_id       1d int-array of trial ids in data
        data           3d numpy array of trials x neurons x sequences
        '''
        
        # Instance variables
        
        assert experiment in [i+1 for i in range(4)], 'experiment must be in range 1-4'
        self.experiment = experiment
        self.train = train
        self.seq_len = seq_len
        self.neur_count = neur_count
        assert seq_start_mode in {'all', 'unique'}, 'seq_start_mode must be in {all, unique}'
        self.seq_start_mode = seq_start_mode
        self.transform = transform if transform else lambda x:x
        self.dataset = None
        self.trail_ids = None
        self.neuron_ids = None
        self.N_sequences = None
        
        if experiment == 1:
            if not os.path.isdir("data/000128"):
                print("Downloading data")
                os.system('dandi download https://dandiarchive.org/dandiset/000128/draft -o data/')

            if train:
                dataset = NWBDataset("data/000128/sub-Jenkins/", "*train", split_heldout=False)
            else:
                dataset = NWBDataset("data/000128/sub-Jenkins/", "*test", split_heldout=False)

        elif experiment == 2:
            if not os.path.isdir("data/000129"):
                print("Downloading data")
                os.system('dandi download https://dandiarchive.org/dandiset/000129/draft -o data/')

            if train:
                dataset = NWBDataset("data/000129/sub-Indy", "*train", split_heldout=False)
            else:
                dataset = NWBDataset("data/000129/sub-Indy", "*test", split_heldout=False)

        elif experiment == 3:
            if not os.path.isdir("data/000127"):
                print("Downloading data")
                os.system('dandi download https://dandiarchive.org/dandiset/000127 -o data/')

            if train:
                dataset = NWBDataset("data/000127/sub-Han/", "*train", split_heldout=False)
            else:
                dataset = NWBDataset("data/000129/sub-Indy", "*test", split_heldout=False)

        elif experiment == 4:
            if not os.path.isdir("data/000130"):
                print("Downloading data")
                os.system('dandi download https://dandiarchive.org/dandiset/000130/draft -o data/')

            if train:
                dataset = NWBDataset("data/000130/sub-Haydn/", "*train", split_heldout=False)
            else:
                dataset = NWBDataset("data/000129/sub-Indy", "*test", split_heldout=False)

        
        # Resample data
        dataset.resample(resample_val)

        # Smooth spikes with 50 ms std Gaussian
        dataset.smooth_spk(50, name='smth_50')

        self.dataset = dataset.make_trial_data()
        
        self.trial_ids = np.unique(self.dataset['trial_id'])
        eligible_trials = {} # ID -> size
        
        max_neur_count = len(self.dataset['spikes'])
        assert max_neur_count >= self.neur_count, f'decrease neuron count, max: {max_neur_count}'
        # Only include trials of suffecient length
        for ID in self.trial_ids:
            trial_len = len(self.dataset[self.dataset['trial_id'] == ID])
            if trial_len >= self.seq_len:
                eligible_trials[ID] = trial_len
        
        assert len(eligible_trials) > 0, 'no eligible trials, decrease sequence length'

        self.trial_ids = np.array(list(eligible_trials.keys()))
        if shuffle: np.random.shuffle(self.trial_ids)

        cur_loc = 0
        possible_starts = [] # list of (ID, time) tuples
        if self.seq_start_mode == 'all':
            # allows for all possible sequences when all seq_len trials elems are from the same trial
            for ID in self.trial_ids:
                lt = [(ID,t) for t in range(eligible_trials[ID] - self.seq_len)]
                possible_starts.extend(lt)
        
        elif self.seq_start_mode == 'unique':
            # creates sequences where each unique elem is in at most one sequence
            for ID in self.trial_ids: 
                lt = [(ID,t) for t in range(0,eligible_trials[ID] - self.seq_len, self.seq_len)]
                possible_starts.extend(lt)
        
        self.possible_starts = possible_starts
        self.N_sequences = len(possible_starts)
        
        self.neuron_ids = np.array(self.dataset['spikes'].keys().tolist())
        np.random.shuffle(self.neuron_ids)

        if neur_count == 0:
            self.neur_count = len(self.neuron_ids)

        self.neuron_ids = self.neuron_ids[:neur_count]

    def __getitem__(self, index):

        trial_id, start = self.possible_starts[index]
        trial_id = np.array([trial_id])
        data = self.transform(self.dataset[trial_id]['spikes'][self.neuron_ids][start:start + self.seq_len])

        return data, self.neuron_ids, trial_id
    
    def __len__(self):
        return self.N_sequences


if __name__ == '__main__':
    nwb_train = NWB(experiment=1, train=True, resample_val=5,
                    seq_len=10, neur_count = 100)





