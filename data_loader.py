import os
import json
import numpy as np
import pandas as pd
import torch.utils.data as data
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.chop import *



class NWB(data.Dataset):
    def __init__(self, experiment, train, resample_val, seq_len, neur_count, shuffle=False, seq_start_mode='all', transform=None):

        '''
        INPUT
        experiment     int from 1-4 chooses experiment (1: MC_Maze_Medium, 2: MC_RTT, 3: Area2_Bump, 4: DMFC_RSG)
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
        
        # Experiment meta info

        EXP_STR = {1: ('000139','sub-Jenkins'), 2: ('000129','sub-Indy'), 3: ('000127','sub-Han'), 4: ('000130', 'sub-Hadyn')}
        
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

        spk_drop_col = [('spikes', spk) for spk in self.neuron_ids[neur_count:]]
        self.neuron_ids = self.neuron_ids[:neur_count]
        dataset.data.drop(spk_drop_col, axis=1, inplace=True)

        print(f'neuron IDs: {self.neuron_ids}')

        # Resample data
        print("Resampling...")
        dataset.resample(resample_val)

        # Smooth spikes with 50 ms std Gaussian
        print("Smoothing...")
        dataset.smooth_spk(50, name='smth_50')
        print('1')
        self.dataset = dataset.make_trial_data()
        print('2')
        self.trial_ids = np.unique(self.dataset['trial_id'])
        eligible_trials = {} # ID -> size
        ineligible_trials = [] # ID
        print('3')
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
        
        self.possible_starts = possible_starts
        self.N_sequences = len(possible_starts)
        

    def __getitem__(self, index):

        trial_id, start = self.possible_starts[index]
        data  = self.dataset[self.dataset['trial_id'] == trial_id]['spikes_smth_50'][self.neuron_ids][start:start + self.seq_len].to_numpy()
        data = self.transform(data) 
        return data, self.neuron_ids, trial_id
    
    def __len__(self):
        return self.N_sequences


if __name__ == '__main__':
    seed = 128

    #manual seed
    np.random.seed(seed)
    # torch.manual_seed(seed)
    nwb_train = NWB(experiment=1, train=True, resample_val=5,
                    seq_len=10, neur_count = 100)
    print(len(nwb_train))
    print(nwb_train[5])




