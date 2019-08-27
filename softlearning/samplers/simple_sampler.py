from collections import defaultdict
import time

import numpy as np

from .base_sampler import BaseSampler


class SimpleSampler(BaseSampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._current_path = defaultdict(list)
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0

    def _process_observations(self,
                              observation,
                              action,
                              mean_action,
                              reward,
                              terminal,
                              next_observation,
                              info,
                              img_dim=(0,)):
        processed_observation = {
            'observations': observation[np.product(img_dim):],
            'images': (observation[:np.product(img_dim)]*255).astype('uint8'),
            'actions': action,
            'mean_actions': mean_action,
            'rewards': [reward],
            'terminals': [terminal],
            'next_observations': next_observation[np.product(img_dim):],
            'next_images': (next_observation[:np.product(img_dim)] * 255).astype('uint8'),
            'infos': info,
        }

        return processed_observation

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        action = self.policy.actions_np([
            self.env.convert_to_active_observation(
                self._current_observation)[None]
        ])[0]

        next_observation, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        processed_sample = self._process_observations(
            observation=self._current_observation,
            action=action,
            mean_action=None,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            info=info,
            img_dim=(4, 256, 256, 1)
        )
        self.policy.ob_rms.update(np.expand_dims(self._current_observation, axis=0))
        self.pool.add_samples(processed_sample)
        for key, value in processed_sample.items():
            self._current_path[key].append(value)

        if terminal or self._path_length >= self._max_path_length:
            last_path = {
                field_name: np.array(values)
                for field_name, values in self._current_path.items()
            }
            #self.pool.add_path(last_path)
            self._last_n_paths.appendleft(last_path)

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self.policy.reset()
            self._current_observation = None
            self._path_length = 0
            self._path_return = 0
            self._current_path = defaultdict(list)

            self._n_episodes += 1
        else:
            self._current_observation = next_observation

        return next_observation, reward, terminal, info

    def random_batch(self, batch_size=None, **kwargs):
        batch_size = batch_size or self._batch_size
        observation_keys = getattr(self.env, 'observation_keys', None)
        start = time.time()
        batch = self.pool.random_batch(
            batch_size, observation_keys=observation_keys, **kwargs)
        print("pool batch time", time.time() - start)
        start = time.time()
        #batch['observations'] = np.hstack([batch['images'].astype('float32')/255, batch['observations']])
        #self.policy.ob_rms.update(batch['observations'])
        #batch['observations'] = self.policy.rms_fn([batch['observations']])[0]
        #batch['next_observations'] = np.hstack([batch['next_images'].astype('float32')/255, batch['next_observations']])
        #batch['next_observations'] = self.policy.rms_fn([batch['next_observations']])[0]
        print("batch processing", time.time() - start)
        #del batch['images']
        #del batch['next_images']
        return batch

    def get_diagnostics(self):
        diagnostics = super(SimpleSampler, self).get_diagnostics()
        diagnostics.update({
            'max-path-return': self._max_path_return,
            'last-path-return': self._last_path_return,
            'episodes': self._n_episodes,
            'total-samples': self._total_samples,
        })

        return diagnostics
