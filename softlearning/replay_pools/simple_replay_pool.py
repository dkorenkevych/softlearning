from collections import defaultdict

import numpy as np
from gym.spaces import Box, Dict, Discrete
from multiprocessing import Process, Array, Value, Lock
from rl_experiments.data_collection.scan_data_db import DBManager
import time
from .flexible_replay_pool import FlexibleReplayPool


def normalize_observation_fields(observation_space, name='observations'):
    if isinstance(observation_space, Dict):
        fields = [
            normalize_observation_fields(child_observation_space, name)
            for name, child_observation_space
            in observation_space.spaces.items()
        ]
        fields = {
            'observations.{}'.format(name): value
            for field in fields
            for name, value in field.items()
        }
    elif isinstance(observation_space, (Box, Discrete)):
        fields = {
            name: {
                'shape': observation_space.shape,
                'dtype': observation_space.dtype,
            }
        }
    else:
        raise NotImplementedError(
            "Observation space of type '{}' not supported."
            "".format(type(observation_space)))

    return fields


class SimpleReplayPool(FlexibleReplayPool):
    def __init__(self, observation_space, action_space, batch_size, prefetch_size=20, img_dim=(0,), db_manager=None, *args, **kwargs):
        self._observation_space = observation_space
        self._action_space = action_space
        self.img_dim = img_dim
        self.db = db_manager
        self.prefetch_size = prefetch_size
        observation_fields = normalize_observation_fields(observation_space)
        #observation_fields['observations']['shape'] = (observation_space.shape[-1] - np.product(self.img_dim),)
        observation_fields['observations']['ctype'] = 'f'
        observation_fields['images'] = {'dtype': 'uint8', 'ctype': 'B', 'shape': (np.product(self.img_dim),)}
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have
        # to worry about termination conditions.
        observation_fields.update({
            'next_' + key: value
            for key, value in observation_fields.items()
        })

        fields = {
            **observation_fields,
            **{
                'actions': {
                    'shape': self._action_space.shape,
                    'dtype': 'float32',
                    'ctype': 'f'
                },
                'rewards': {
                    'shape': (1, ),
                    'dtype': 'float32',
                    'ctype': 'f'
                },
                # self.terminals[i] = a terminal was received at time i
                'terminals': {
                    'shape': (1, ),
                    'dtype': 'bool',
                    'ctype': 'B'
                },
            }
        }
        self.field_params = fields
        self.batch_data = {
                'rewards': [],
                'actions': [],
                'terminals': [],
                "observations": [],
                "next_observations": [],
                # "images": [],
                # "next_images": []
        }
        for modality in self.batch_data.keys():
            for _ in range(self.prefetch_size):
                self.batch_data[modality].append(np.frombuffer(Array(fields[modality]['ctype'],
                                                                     int(batch_size * np.product(fields[modality]['shape']))).get_obj(),
                                                       dtype=fields[modality]['dtype']).reshape((batch_size, ) + fields[modality]['shape']))
        super(SimpleReplayPool, self).__init__(
            *args, fields_attrs=fields, **kwargs)

        self.read_buffer_p = Value('i', 0)
        self.write_buffer_p = Value('i', 0)
        self.fresh_data = Value('i', 0)
        self._access_lock = Lock()
        self._fetch_loop = Process(target=self.fetch_loop, args=(batch_size,))

    def fetch_loop(self, batch_size, field_name_filter=None):
        db = DBManager(table=self.db.table)
        while True:
            if self.db.last_id < 1000:
                time.sleep(10)
                continue
            try:
                if self.fresh_data.value < self.prefetch_size:
                    #time.sleep(0.5)
                    print("prefetching a batch", self.fresh_data.value)
                    #random_indices = np.random.choice(range(np.minimum(db.last_id, 290000)), batch_size, replace=False)
                    random_indices = np.random.choice(range(db.last_id - 1000000, db.last_id), batch_size, replace=False)
                    data = self.batch_by_indices_db(db,
                                             random_indices, field_name_filter=field_name_filter)
                    data_batch = data['images']
                    orig_shape = data_batch.shape
                    data_batch = data_batch.reshape((data['images'].shape[0], 4, 256, 256))
                    num_im = data_batch.shape[0] * data_batch.shape[1]
                    shift_x = np.random.randint(-10, 10, num_im)
                    shift_y = np.random.randint(-10, 10, num_im)
                    new_batch = np.zeros(data_batch.shape).reshape((num_im,) + data_batch.shape[2:])
                    big_im = np.zeros(
                        (num_im,) + (data_batch.shape[2] + 20, data_batch.shape[3] + 20) + data_batch.shape[4:])
                    data_batch = data_batch.reshape(new_batch.shape)
                    for i in range(num_im):
                        big_im[i, 10 + shift_x[i]: 10 + shift_x[i] + data_batch.shape[1],
                        10 + shift_y[i]:10 + shift_y[i] + data_batch.shape[2]] = data_batch[i]
                    new_batch = big_im[:, 10:-10, 10:-10].reshape(orig_shape)

                    #new_batch_next = np.zeros(data_batch.shape).reshape((num_im,) + data_batch.shape[2:])
                    big_im = np.zeros(
                        (num_im,) + (data_batch.shape[1] + 20, data_batch.shape[2] + 20))
                    data_batch = data['next_images'].reshape(data_batch.shape)
                    for i in range(num_im):
                        big_im[i, 10 + shift_x[i]: 10 + shift_x[i] + data_batch.shape[1],
                        10 + shift_y[i]:10 + shift_y[i] + data_batch.shape[2]] = data_batch[i]
                    new_batch_next = big_im[:, 10:-10, 10:-10].reshape(orig_shape)

                    data['observations'] = np.hstack([new_batch.astype('float32') / 255, data['observations']])
                    data['next_observations'] = np.hstack(
                        [new_batch_next.astype('float32') / 255, data['next_observations']])
    
                    # data['observations'] = np.hstack([data['images'].astype('float32') / 255, data['observations']])
                    # data['next_observations'] = np.hstack(
                    #     [data['next_images'].astype('float32') / 255, data['next_observations']])
                    buffer_pointer = self.write_buffer_p.value
                    del data['images']
                    del data['next_images']
                    for modality in data:
                        np.copyto(self.batch_data[modality][buffer_pointer], data[modality].astype(self.field_params[modality]['dtype']))
                    self.write_buffer_p.value = (buffer_pointer + 1) % self.prefetch_size
                    self._access_lock.acquire()
                    self.fresh_data.value += 1
                    self._access_lock.release()
                time.sleep(0.1)
            except Exception as e:
                print("Couldn't get new data, reconnecting", e)
                try:
                    db.cnx, db.cursor = db.make_connection()
                except Exception as e:
                    print("Couldn't reconnect to db", e)

                time.sleep(1.0)



    def add_samples_base(self, samples):
        field_names = list(samples.keys())
        num_samples = samples[field_names[0]].shape[0]

        index = np.arange(
            self._pointer, self._pointer + num_samples) % self._max_size

        for field_name in self.field_names:
            default_value = (
                self.fields_attrs[field_name].get('default_value', 0.0))
            values = samples.get(field_name, default_value)
            assert values.shape[0] == num_samples
            self.fields[field_name][index] = values

    def random_batch_old(self, batch_size, field_name_filter=None, **kwargs):
        random_indices = np.random.choice(range(self.db.last_id), batch_size, replace=False)
        return self.batch_by_indices_db(self.db,
            random_indices, field_name_filter=field_name_filter, **kwargs)

    def random_batch(self, batch_size, field_name_filter=None, **kwargs):
        while True:
            if self.fresh_data.value <= 0:
                print("No fresh data available, waiting", self.fresh_data.value)
                time.sleep(0.1)
                continue
            buffer_pointer = self.read_buffer_p.value
            batch_data = {}
            for modality in self.batch_data:
                batch_data[modality] = self.batch_data[modality][buffer_pointer].copy()
            self.read_buffer_p.value = (buffer_pointer + 1) % self.prefetch_size
            self._access_lock.acquire()
            self.fresh_data.value -= 1
            self._access_lock.release()
            return batch_data


    def add_samples(self, samples):
        if not isinstance(self._observation_space, Dict):
            self.db.save_samples(samples)
            return
            return super(SimpleReplayPool, self).add_samples(samples)

        dict_observations = defaultdict(list)
        for observation in samples['observations']:
            for key, value in observation.items():
                dict_observations[key].append(value)

        dict_next_observations = defaultdict(list)
        for next_observation in samples['next_observations']:
            for key, value in next_observation.items():
                dict_next_observations[key].append(value)

        samples.update(
           **{
               f'observations.{observation_key}': np.array(values)
               for observation_key, values in dict_observations.items()
           },
           **{
               f'next_observations.{observation_key}': np.array(values)
               for observation_key, values in dict_next_observations.items()
           },
        )

        del samples['observations']
        del samples['next_observations']

        return super(SimpleReplayPool, self).add_samples(samples)

    def batch_by_indices(self, indices, field_name_filter=None):
        if np.any(indices % self._max_size > self.size):
            raise ValueError(
                "Tried to retrieve batch with indices greater than current"
                " size")

        field_names = self.field_names
        if field_name_filter is not None:
            field_names = self.filter_fields(
                field_names, field_name_filter)

        return {
            field_name: self.fields[field_name][indices]
            for field_name in field_names
        }

    def batch_by_indices_db(self,
                         db,
                         indices,
                         field_name_filter=None,
                         observation_keys=None):
        data = db.get_samples_by_id(indices)

        field_names = self.field_names
        if field_name_filter is not None:
            field_names = self.filter_fields(
                field_names, field_name_filter)

        return {
            field_name: np.stack(data[field_name])
            for field_name in field_names
        }


    def batch_by_indices(self,
                         indices,
                         field_name_filter=None,
                         observation_keys=None):
        if not isinstance(self._observation_space, Dict):
            return super(SimpleReplayPool, self).batch_by_indices(
                indices, field_name_filter=field_name_filter)

        batch = {
            field_name: self.fields[field_name][indices]
            for field_name in self.field_names
        }

        if observation_keys is None:
            observation_keys = tuple(self._observation_space.spaces.keys())

        observations = np.concatenate([
            batch['observations.{}'.format(key)]
            for key in observation_keys
        ], axis=-1)

        next_observations = np.concatenate([
            batch['next_observations.{}'.format(key)]
            for key in observation_keys
        ], axis=-1)

        batch['observations'] = observations
        batch['next_observations'] = next_observations

        if field_name_filter is not None:
            filtered_fields = self.filter_fields(
                batch.keys(), field_name_filter)
            batch = {
                field_name: batch[field_name]
                for field_name in filtered_fields
            }

        return batch

    def terminate_episode(self):
        pass
