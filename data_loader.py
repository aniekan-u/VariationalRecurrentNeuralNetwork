import os
import warnings
import random
from copy import copy, deepcopy
import numpy as np
import pandas as pd
import torch.utils.data as data
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.chop import *
import torch



class NWB(data.Dataset):
    def __init__(self, experiment, train, resample_val, seq_len, neur_count, N_seq, parts_fract_seq=None, shuffle=False, seq_start_mode='all', transform=None):

        '''
        INPUT
        experiment       int from 1-4 chooses experiment (1: MC_Maze_Medium, 2: MC_RTT, 3: Area2_Bump, 4: DMFC_RSG)
        train            bool true -> train, false -> test
        resample_val     int determines factor of resampling,    1 returns original, once resampled need to redownload to get original (delete folder)
        seq_len          int length of individual sequences
        neur_count       int count of neurons per trial,         0 for full length
        N_seq            int total number of sequences           0 for max
        parts_fract_seq  dict maps partition -> fract            0-1
        seq_start_mode   str                                     ['all', 'unique']
        tranform         function

        OUTPUT
        neuron_id        2d int-array of neuron ids in data
        trial_id         1d int-array of trial ids in data
        data             3d numpy array of trials x neurons x sequences
        '''

        # Experiment meta info

        EXP_STR = {1: ('000139','sub-Jenkins'), 2: ('000129','sub-Indy'), 3: ('000127','sub-Han'), 4: ('000130', 'sub-Hadyn')}

        # Instance variables

        assert experiment in [i+1 for i in range(4)], 'experiment must be in range 1-4'
        self.experiment = experiment
        self.train = train
        self.seq_len = seq_len
        self.neur_count = neur_count
        if parts_fract_seq is None:
            self.parts_fract_seq = {'default': 1.}
            self.curr_part = 'default'
        else:
            assert sum(parts_fract_seq.values()) == 1., 'partitions must sum to 1.'
            self.parts_fract_seq = parts_fract_seq
            self.curr_part = None
        assert seq_start_mode in {'all', 'unique'}, 'seq_start_mode must be in {all, unique}'
        self.seq_start_mode = seq_start_mode
        self.transform = transform if transform else lambda x:x
        self.dataset = None
        self.trail_ids = None
        self.neuron_ids = None
        self.N_sequences = None

        # Columns to drop
        drop_col = ['cursor_pos', 'eye_pos', 'hand_pos', 'hand_vel']

        print("Getting Dataset....")
        save_path = 'data/' + EXP_STR[experiment][0] + '/'
        data_path = save_path + EXP_STR[experiment][1] + '/'
        dandi_path = EXP_STR[experiment][0] + '/draft'

        if not os.path.isdir(save_path):
            print("Downloading data")
            os.system('dandi download https://dandiarchive.org/dandiset/' + dandi_path + ' -o data/')

        if self.train:
            dataset = NWBDataset(data_path, "*train", split_heldout=False, skip_fields=drop_col)
        else:
            dataset = NWBDataset(data_path, "*test", split_heldout=False, skip_fields=drop_col)

        # Picking subset of spikes
        self.neuron_ids = np.array(dataset.data['spikes'].keys().tolist())
        np.random.shuffle(self.neuron_ids)

        if neur_count == 0:
            self.neur_count = len(self.neuron_ids)

        spk_drop_col = [('spikes', spk) for spk in self.neuron_ids[self.neur_count:]]
        self.neuron_ids = self.neuron_ids[:self.neur_count]
        dataset.data.drop(spk_drop_col, axis=1, inplace=True)

        print(f'neuron IDs: {self.neuron_ids}')

        # Resample data
        print("Resampling...")
        dataset.resample(resample_val)

        # Smooth spikes with 50 ms std Gaussian
        print("Smoothing...")
        dataset.smooth_spk(50, name='smth_50')
        
        self.dataset = dataset.make_trial_data()
        
        self.trial_ids = np.unique(self.dataset['trial_id'])
        eligible_trials = {} # ID -> size
        ineligible_trials = [] # ID
        max_neur_count = len(self.dataset['spikes'])
        assert max_neur_count >= self.neur_count, f'decrease neuron count, max: {max_neur_count}'

        # Only include trials of suffecient length
        print("Finding eligible trials...")
        for ID in self.trial_ids:
            trial_len = len(self.dataset[self.dataset['trial_id'] == ID])
            if trial_len >= self.seq_len:
                eligible_trials[ID] = trial_len
            else:
                ineligible_trials.append(ID)

        assert len(eligible_trials) > 0, 'no eligible trials, decrease sequence length'

        # Drop ineligible trials
        for tr in ineligible_trials:
            self.dataset.drop(self.dataset[self.dataset.trial_id == tr].index, inplace=True)

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
        
        if shuffle: random.shuffle(possible_starts)
        self.possible_starts = {}
        if N_seq > len(possible_starts) or N_seq == 0:
            N_seq = len(possible_starts)
            warnings.warn(f'Dataset only has {N_seq} possible sequences')

        self.N_sequences = 0
        start = 0
        for part, fract in self.parts_fract_seq.items():
            n_seq = int(fract * N_seq)
            self.N_sequences += n_seq
            self.possible_starts[part] = possible_starts[start:start + n_seq]
            start += n_seq
    
    def set_curr_part(self, part):
        assert part in self.parts_fract_seq.keys(), 'Invalid partition'
        self.curr_part = part
    
    def get_total_num_seq(self):
        return self.N_sequences

    def __getitem__(self, index):
        assert self.curr_part is not None, 'Set the current partition'
        trial_id, start = self.possible_starts[self.curr_part][index]
        data = self.dataset[self.dataset['trial_id'] == trial_id]
        data = data['spikes_smth_50'][self.neuron_ids]
        data = self.transform(data[start:start + self.seq_len].to_numpy())
        return data, self.neuron_ids, trial_id
    
    def __len__(self):
        return len(self.possible_starts[self.curr_part])

    def __copy__(self):
        cls = self.__class__
        obj = cls.__new__(cls)
        obj.__dict__.update(self.__dict__)
        return obj
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        obj = cls.__new__(cls)
        memo[id(self)] = obj
        for k, v in self.__dict__.items():
            setattr(obj, k, deepcopy(v, memo))
        return obj

if __name__ == '__main__':
    seed = 128

    #manual seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    parts = {'train': .8, 'val': .2}
    nwb_train = NWB(experiment=1, train=True, resample_val=5, seq_len=10, neur_count = 100,
                    N_seq=100, parts_fract_seq=parts, shuffle=True,  seq_start_mode='unique')
    
    nwb_train.set_curr_part('val')
    print(nwb_train.possible_starts)
    print(nwb_train[1])
    print(len(nwb_train))
