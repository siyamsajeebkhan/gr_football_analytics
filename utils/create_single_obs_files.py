### !!!!!!!!!!!!!!!!!!! Do not delete this cell !!!!!!!!!!!!!!!!!!!
import os
from tqdm import tqdm
import pickle
import time
import numpy as np
from gfootball.env.wrappers import Simple115StateWrapper

# Here we will write the pickle files
def prepare_npy_dataset_from_replay_files(replay_files, replay_files_path):
    obs_save_dir = '/home/ssk/Study/GRP/dataset/npy_files'
    # replay_files_path = 'dataset/replay_files'

    if not os.path.exists(obs_save_dir):
        os.mkdir(obs_save_dir)

    for replay in tqdm(replay_files):
        with open(os.path.join(replay_files_path, replay), 'rb') as pkl_file:
            episode_data = pickle.load(pkl_file)

        episode_no = replay.split('.')[0]
        episode = episode_data['observations']
        episode['active'] = episode_data['players'][0]['active']
        episode_length = 3002
        raw_obs = {}

        episode_dir = os.path.join(obs_save_dir, episode_no)
        if not os.path.exists(episode_dir):
            os.mkdir(episode_dir)

        for step in range(episode_length):
            for (key, item) in episode.items():
                raw_obs[key] = item[step]

            float115_frame =  Simple115StateWrapper.convert_observation([raw_obs], True)[0].tolist()
            action = episode_data['players'][0]['action'][step]
            
            frame_name = episode_no+f'_{step}'
            if len(action) != 0:
                float115_frame.extend(action)
                fram_save_path = os.path.join(episode_dir, frame_name)
                np.save(fram_save_path, np.array(float115_frame))



if __name__=='__main__':
    replay_files_path = 'dataset/replay_files'
    replay_files = sorted(os.listdir(replay_files_path))
    replay_files.pop(0)
    # replay_files = replay_files[0:1]
    print(f"total replay files: {len(replay_files)}")
    # replay_files = replay_files[0:1]

    start = time.perf_counter()
    prepare_npy_dataset_from_replay_files(replay_files, replay_files_path)
    end =  time.perf_counter()

    print(f"Total time needed to process {len(replay_files)}: {end-start}s")
    print(f"Time needed to process a single file: {(end-start)/len(replay_files)}s")