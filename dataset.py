import os
import numpy as np
import itertools
import collections
import torch
from .example import Example
from .utils import nostdout
from pycocotools.coco import COCO as pyCOCO
import json


class Dataset(object):
    def __init__(self, examples, fields):
        self.examples = examples
        self.fields = dict(fields)

    def collate_fn(self):
        def collate(batch):

            if len(self.fields) == 1:
                batch = [batch, ]
            else:
                batch = list(zip(*batch))

            tensors = []
            for field, data in zip(self.fields.values(), batch):
                tensor = field.process(data)
                if isinstance(tensor, collections.Sequence) and any(isinstance(t, torch.Tensor) for t in tensor):
                    tensors.extend(tensor)
                else:
                    tensors.append(tensor)


            if len(tensors) == 3:
                a = []
                a.append(tensors[0])
                a.append(tensors[1])
                a.append(tensors[2])
                return a
            if len(tensors) == 2:
                return tensors
            else:
                return tensors[0]

        return collate


    def __getitem__(self, i):
        example = self.examples[i]
        data = []
        for field_name, field in self.fields.items():
            # data.append(field.preprocess(getattr(example,field_name)))
            tem = getattr(example, field_name)
            data.append(field.preprocess(tem))

        if len(data) == 1:
            if field_name == 'image':
                data = data[0]
            elif field_name == 'text':
                data = data[0]
        return data

    def __len__(self):
        return len(self.examples)

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)


class ValueDataset(Dataset):
    def __init__(self, examples, fields, dictionary):
        self.dictionary = dictionary
        super(ValueDataset, self).__init__(examples, fields)

    def collate_fn(self):
        def collate(batch):
            value_batch_flattened = list(itertools.chain(*batch))
            value_tensors_flattened = super(ValueDataset, self).collate_fn()(value_batch_flattened)

            lengths = [0, ] + list(itertools.accumulate([len(x) for x in batch]))
            if isinstance(value_tensors_flattened, collections.Sequence) \
                    and any(isinstance(t, torch.Tensor) for t in value_tensors_flattened):
                value_tensors = [[vt[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])] for vt in
                                 value_tensors_flattened]
            else:
                value_tensors = [value_tensors_flattened[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])]

            return value_tensors

        return collate

    def __getitem__(self, i):
        if i not in self.dictionary:
            raise IndexError

        values_data = []
        for idx in self.dictionary[i]:
            value_data = super(ValueDataset, self).__getitem__(idx)
            values_data.append(value_data)
        return values_data

    def __len__(self):
        return len(self.dictionary)


class DictionaryDataset(Dataset):
    def __init__(self, examples, fields, key_fields):
        if not isinstance(key_fields, (tuple, list)):
            key_fields = (key_fields,)
        for field in key_fields:
            assert (field in fields)

        dictionary = collections.defaultdict(list)
        key_fields = {k: fields[k] for k in key_fields}
        value_fields = {k: fields[k] for k in fields.keys() if k not in key_fields}
        key_examples = []
        key_dict = dict()
        value_examples = []

        for i, e in enumerate(examples):  # examples是list，每个元素是{image：text}
            key_example = Example.fromdict({k: getattr(e, k) for k in key_fields})
            value_example = Example.fromdict({v: getattr(e, v) for v in value_fields})
            if key_example not in key_dict:
                key_dict[key_example] = len(key_examples)
                key_examples.append(key_example)

            value_examples.append(value_example)
            dictionary[key_dict[key_example]].append(i)

        self.key_dataset = Dataset(key_examples, key_fields)
        self.value_dataset = ValueDataset(value_examples, value_fields, dictionary)
        super(DictionaryDataset, self).__init__(examples, fields)

    def collate_fn(self):
        def collate(batch):
            key_batch, value_batch = list(zip(*batch))
            key_tensors = self.key_dataset.collate_fn()(key_batch)
            value_tensors = self.value_dataset.collate_fn()(value_batch)
            return key_tensors, value_tensors

        return collate

    def __getitem__(self, i):
        return self.key_dataset[i], self.value_dataset[i]

    def __len__(self):
        return len(self.key_dataset)


def unique(sequence):
    seen = set()
    if isinstance(sequence[0], list):
        return [x for x in sequence if not (tuple(x) in seen or seen.add(tuple(x)))]
    else:
        return [x for x in sequence if not (x in seen or seen.add(x))]


class PairedDataset(Dataset):
    def __init__(self, examples, fields):
        assert ('image' in fields)
        assert ('text' in fields)
        super(PairedDataset, self).__init__(examples, fields)
        self.image_field = self.fields['image']
        self.text_field = self.fields['text']

    def image_set(self):
        img_list = [e.image for e in self.examples]
        image_set = unique(img_list)
        examples = [Example.fromdict({'image': i}) for i in image_set]
        dataset = Dataset(examples, {'image': self.image_field})
        return dataset

    def text_set(self):
        text_list = [e.text for e in self.examples]
        text_list = unique(text_list)
        examples = [Example.fromdict({'text': t}) for t in text_list]
        dataset = Dataset(examples, {'text': self.text_field})
        return dataset

    def image_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = DictionaryDataset(self.examples, fields, key_fields='image')
        return dataset

    def text_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = DictionaryDataset(self.examples, fields, key_fields='text')
        return dataset

    @property
    def splits(self):
        raise NotImplementedError



class Sydney(PairedDataset):
    def __init__(self, image_field, text_field, img_root, ann_root, id_root=None, use_restval=True,
                 cut_validation=False):
        roots = {}
        img_root = os.path.join(img_root, '')
        roots['train'] = {
            'img': os.path.join(img_root, ''),
            'cap': os.path.join(ann_root, 'dataset.json')
        }
        roots['val'] = {
            'img': os.path.join(img_root, ''),
            'cap': os.path.join(ann_root, 'dataset.json')
        }
        roots['test'] = {
            'img': os.path.join(img_root, ''),
            'cap': os.path.join(ann_root, 'dataset.json')
        }
        ids = {'train': None, 'val': None, 'test': None}

        with nostdout():  # dataset处理后，每个example自带编号(每个数据集都从00000开始)，都是image绝对路径+一个caption
            self.train_examples, self.val_examples, self.test_examples = self.get_samples(img_root, ann_root,
                                                                                          ids)
        examples = self.train_examples + self.val_examples + self.test_examples  # 三个数据集合在一起
        super(Sydney, self).__init__(examples, {'image': image_field, 'text': text_field})  # 调用PairedDataset

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
        return train_split, val_split, test_split

    @classmethod
    def get_samples(cls, img_root, ann_root, ids_dataset=None):
        train_samples = []
        val_samples = []
        test_samples = []
        dataset = json.load(open(os.path.join(ann_root, 'dataset.json'), 'r'))['images']
        index = list(range(613))
        np.random.shuffle(index)
        a1 = index[:497]
        a2 = index[497:555]
        a3 = index[555:613]
        for i, d in enumerate(dataset):
            if i in a1:
                for x in range(len(d['sentences'])):
                    captions = d['sentences'][x]['raw']
                    filename = d['filename']
                    # imgid = d['imgid']
                    example = Example.fromdict({'image': os.path.join(img_root, str(filename)), 'text': captions})
                    train_samples.append(example)
            elif i in a2:
                for x in range(len(d['sentences'])):
                    captions = d['sentences'][x]['raw']
                    filename = d['filename']
                    example = Example.fromdict({'image': os.path.join(img_root, str(filename)), 'text': captions})
                    val_samples.append(example)
            elif i in a3:
                for x in range(len(d['sentences'])):
                    captions = d['sentences'][x]['raw']
                    filename = d['filename']
                    example = Example.fromdict({'image': os.path.join(img_root, str(filename)), 'text': captions})
                    test_samples.append(example)

        return train_samples, val_samples, test_samples



class UCM(PairedDataset):
    def __init__(self, image_field, text_field, img_root, ann_root, id_root=None, use_restval=True,
                 cut_validation=False):

        roots = {}
        img_root = os.path.join(img_root, '')
        roots['train'] = {
            'img': os.path.join(img_root, ''),
            'cap': os.path.join(ann_root, 'dataset.json')
        }
        roots['val'] = {
            'img': os.path.join(img_root, ''),
            'cap': os.path.join(ann_root, 'dataset.json')
        }
        roots['test'] = {
            'img': os.path.join(img_root, ''),
            'cap': os.path.join(ann_root, 'dataset.json')
        }
        ids = {'train': None, 'val': None, 'test': None}

        with nostdout():  # dataset处理后，每个example自带编号(每个数据集都从00000开始)，都是image绝对路径+一个caption
            self.train_examples, self.val_examples, self.test_examples = self.get_samples(img_root, ann_root, ids)
        examples = self.train_examples + self.val_examples + self.test_examples  # 三个数据集合在一起
        super(UCM, self).__init__(examples, {'image': image_field, 'text': text_field})  # 调用PairedDataset

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
        return train_split, val_split, test_split

    @classmethod
    def get_samples(cls, img_root, ann_root, ids_dataset=None):
        train_samples = []
        val_samples = []
        test_samples = []

        # dataset = jsonmod.load(open(json, 'r'))['images']
        dataset = json.load(open(os.path.join(ann_root, 'dataset.json'), 'r'))['images']
        index = list(range(2100))
        np.random.shuffle(index)
        a1 = index[:1680]
        a2 = index[1680:1890]
        a3 = index[1890:2100]
        for i, d in enumerate(dataset):
            if i in a1:
                for x in range(len(d['sentences'])):
                    captions = d['sentences'][x]['raw']
                    filename = d['filename']
                    # imgid = d['imgid']
                    example = Example.fromdict({'image': os.path.join(img_root, str(filename)), 'text': captions})
                    train_samples.append(example)
            elif i in a2:
                for x in range(len(d['sentences'])):
                    captions = d['sentences'][x]['raw']
                    filename = d['filename']
                    example = Example.fromdict({'image': os.path.join(img_root, str(filename)), 'text': captions})
                    val_samples.append(example)
            elif i in a3:
                for x in range(len(d['sentences'])):
                    captions = d['sentences'][x]['raw']
                    filename = d['filename']
                    example = Example.fromdict({'image': os.path.join(img_root, str(filename)), 'text': captions})
                    test_samples.append(example)

        return train_samples, val_samples, test_samples



class RSICD(PairedDataset):
    def __init__(self, image_field, text_field, img_root, ann_root, id_root=None, use_restval=True,
                 cut_validation=False):

        roots = {}
        img_root = os.path.join(img_root, '')
        roots['train'] = {
            'img': os.path.join(img_root, ''),
            'cap': os.path.join(ann_root, 'dataset.json')
        }
        roots['val'] = {
            'img': os.path.join(img_root, ''),
            'cap': os.path.join(ann_root, 'dataset.json')
        }
        roots['test'] = {
            'img': os.path.join(img_root, ''),
            'cap': os.path.join(ann_root, 'dataset.json')
        }
        ids = {'train': None, 'val': None, 'test': None}

        with nostdout():
            self.train_examples, self.val_examples, self.test_examples = self.get_samples(img_root, ann_root, ids)
        examples = self.train_examples + self.val_examples + self.test_examples  # 三个数据集合在一起
        super(RSICD, self).__init__(examples, {'image': image_field, 'text': text_field})  # 调用PairedDataset

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
        return train_split, val_split, test_split

    @classmethod
    def get_samples(cls, img_root, ann_root, ids_dataset=None):
        train_samples = []
        val_samples = []
        test_samples = []
        dataset = json.load(open(os.path.join(ann_root, 'dataset_rsicd.json'), 'r'))['images']

        index = list(range(10921))
        np.random.shuffle(index)
        a1 = index[:8734]
        a2 = index[8734:9828]
        a3 = index[9828:10921]
        for i, d in enumerate(dataset):
            if i in a1:
                for x in range(len(d['sentences'])):
                    captions = d['sentences'][x]['raw']
                    filename = d['filename']
                    # imgid = d['imgid']
                    example = Example.fromdict({'image': os.path.join(img_root, str(filename)), 'text': captions})
                    train_samples.append(example)
            elif i in a2:
                for x in range(len(d['sentences'])):
                    captions = d['sentences'][x]['raw']
                    filename = d['filename']
                    example = Example.fromdict({'image': os.path.join(img_root, str(filename)), 'text': captions})
                    val_samples.append(example)
            elif i in a3:
                for x in range(len(d['sentences'])):
                    captions = d['sentences'][x]['raw']
                    filename = d['filename']
                    example = Example.fromdict({'image': os.path.join(img_root, str(filename)), 'text': captions})
                    test_samples.append(example)

        return train_samples, val_samples, test_samples