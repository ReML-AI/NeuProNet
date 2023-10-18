# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import sys
import io

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from .. import FairseqDataset
from ..data_utils import compute_mask_indices, get_buckets, get_bucketed_sizes
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
)
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel


logger = logging.getLogger(__name__)


class STFTAudioDataset(FairseqDataset):
    def __init__(
        self,
        # sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        compute_mask_indices=False,
        num_mels=128,
        profiling=False,
        profile_sort=False,
        wave2graph=False,
        adresso_pretrain=False,
        **mask_compute_kwargs,
    ):
        super().__init__()

        # self.sample_rate = sample_rate
        self.sizes = []
        self.num_mels = num_mels
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize
        self.compute_mask_indices = compute_mask_indices
        self.profiling = profiling
        self.profile_sort = profile_sort
        self.wave2graph = wave2graph
        self.adresso_pretrain = adresso_pretrain
        if self.profiling or self.profile_sort:
            self.profiles = []
        if self.compute_mask_indices:
            self.mask_compute_kwargs = mask_compute_kwargs
            self._features_size_map = {}
            self._C = mask_compute_kwargs["encoder_embed_dim"]
            self._conv_feature_layers = eval(mask_compute_kwargs["conv_feature_layers"])

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    # def postprocess(self, feats, curr_sample_rate):
    #     # if feats.dim() == 2:
    #     #     feats = feats.mean(-1)

    #     # if curr_sample_rate != self.sample_rate:
    #     #     raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

    #     # assert feats.dim() == 1, feats.dim()

    #     if self.normalize:
    #         with torch.no_grad():
    #             feats = F.layer_norm(feats, feats.shape)
    #     return feats

    def crop_to_max_size(self, spec, target_size):
        size = spec.shape[1]
        diff = size - target_size
        if diff <= 0:
            return spec

        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return spec[:, start:end]

    def _compute_mask_indices(self, dims, padding_mask):
        B, T, C = dims
        mask_indices, mask_channel_indices = None, None
        if self.mask_compute_kwargs["mask_prob"] > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_compute_kwargs["mask_prob"],
                self.mask_compute_kwargs["mask_length"],
                self.mask_compute_kwargs["mask_selection"],
                self.mask_compute_kwargs["mask_other"],
                min_masks=2,
                no_overlap=self.mask_compute_kwargs["no_mask_overlap"],
                min_space=self.mask_compute_kwargs["mask_min_space"],
            )
            mask_indices = torch.from_numpy(mask_indices)
        if self.mask_compute_kwargs["mask_channel_prob"] > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_compute_kwargs["mask_channel_prob"],
                self.mask_compute_kwargs["mask_channel_length"],
                self.mask_compute_kwargs["mask_channel_selection"],
                self.mask_compute_kwargs["mask_channel_other"],
                no_overlap=self.mask_compute_kwargs["no_mask_channel_overlap"],
                min_space=self.mask_compute_kwargs["mask_channel_min_space"],
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices).unsqueeze(1).expand(-1, T, -1)
            )

        return mask_indices, mask_channel_indices

    @staticmethod
    def _bucket_tensor(tensor, num_pad, value):
        return F.pad(tensor, (0, num_pad), value=value)

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        profiling = self.profiling
        if len(samples) == 0:
            return {}
        if "profile" not in samples[0].keys():
            profiling = False

        sources = [s["source"] for s in samples]
        sizes = [s.shape[1] for s in sources]
        if profiling:
            profiles = [s["profile"] for s in samples]
            profile_groups = [s["profile_group"] for s in samples]

        if self.wave2graph:
            mfccs = [s['mfcc'] for s in samples]
            corrs = [s['corr'] for s in samples]

            mfcc_sizes = [s.shape[1] for s in mfccs]
            target_size = 800
            if self.adresso_pretrain:
                target_size = 801*5
            collated_mfccs = mfccs[0].new_zeros(len(mfccs), 39, target_size)
            # print(collated_sources.shape, sources[0].shape)
            mfcc_padding_mask = (
                torch.BoolTensor(collated_mfccs.shape).fill_(False)
            )
            for i, (source, size) in enumerate(zip(mfccs, mfcc_sizes)):
                diff = size - target_size
                if diff == 0:
                    collated_mfccs[i] = source
                elif diff < 0:
                    assert self.pad
                    collated_mfccs[i] = torch.cat(
                        [source, source.new_full((39, -diff,), 0.0)],
                        dim=1
                    )
                    mfcc_padding_mask[i, :, diff:] = True
                else:
                    collated_mfccs[i] = self.crop_to_max_size(source, target_size)
            try:
                mean = torch.Tensor(np.load(os.path.join(self.root_dir, 'mean.npy')))
                std = torch.Tensor(np.load(os.path.join(self.root_dir, 'std.npy')))
                collated_mfccs = (collated_mfccs-mean)/std
            except Exception as _:
                # fold = self.manifest_path[:self.manifest_path.rfind('/')][-1]
                fold = self.manifest_path[self.manifest_path.find('fold')+4]
                split = 'train' if 'train' in self.manifest_path else 'valid' if 'valid' in self.manifest_path else 'test'
                mean = torch.Tensor(np.load(os.path.join(self.root_dir, f'mean_{split}_fold{fold}.npy')))
                std = torch.Tensor(np.load(os.path.join(self.root_dir, f'std_{split}_fold{fold}.npy')))
                collated_mfccs = (collated_mfccs-mean)/std

        if self.adresso_pretrain:
            pretrains = [s['pretrain'] for s in samples]
            w2v_pretrains = [s['w2v_pretrain'] for s in samples]
            os_pretrains = [s['os_pretrain'] for s in samples]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(len(sources), self.num_mels, target_size)
        # print(collated_sources.shape, sources[0].shape)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((self.num_mels, -diff,), 0.0)],
                    dim=1
                )
                padding_mask[i, :, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        if profiling:
            input["profile"] = profiles
            input["profile_group"] = profile_groups
        if self.wave2graph:
            # input["mfcc"] = torch.stack(mfccs)
            input["mfcc"] = collated_mfccs
            input["corr"] = torch.stack(corrs)
        if self.adresso_pretrain:
            input["pretrain"] = torch.stack(pretrains)
            input["w2v_pretrain"] = torch.stack(w2v_pretrains)
            input["os_pretrain"] = torch.stack(os_pretrains)
            # import pdb; pdb.set_trace()
            # from sklearn.preprocessing import StandardScaler
            pretrained = torch.concat((input["pretrain"], input["w2v_pretrain"], input["os_pretrain"]), dim=-1)
            fold = self.manifest_path[self.manifest_path.find('fold')+4]
            split = 'train' if 'train' in self.manifest_path else 'valid' if 'valid' in self.manifest_path else 'test'
            import joblib
            scaler = joblib.load(f'/cm/shared/tungtk2/ADReSSo/train/audio/scaler_{split}_fold{fold}.save')
            pretrained = scaler.transform(pretrained)
            input["pretrain"] = torch.Tensor(pretrained[:, :input["pretrain"].shape[1]])
            input["w2v_pretrain"] = torch.Tensor(pretrained[:, input["pretrain"].shape[1]:input["pretrain"].shape[1] + input["w2v_pretrain"].shape[1]])
            input["os_pretrain"] = torch.Tensor(pretrained[:, input["pretrain"].shape[1] + input["w2v_pretrain"].shape[1]:])


        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if self.pad:
            input["padding_mask"] = padding_mask

        if hasattr(self, "num_buckets") and self.num_buckets > 0:
            assert self.pad, "Cannot bucket without padding first."
            bucket = max(self._bucketed_sizes[s["id"]] for s in samples)
            num_pad = bucket - collated_sources.size(-1)
            if num_pad:
                input["source"] = self._bucket_tensor(collated_sources, num_pad, 0)
                input["padding_mask"] = self._bucket_tensor(padding_mask, num_pad, True)

        if self.compute_mask_indices:
            B = input["source"].size(0)
            T = self._get_mask_indices_dims(input["source"].size(-1))
            padding_mask_reshaped = input["padding_mask"].clone()
            extra = padding_mask_reshaped.size(1) % T
            if extra > 0:
                padding_mask_reshaped = padding_mask_reshaped[:, :-extra]
            padding_mask_reshaped = padding_mask_reshaped.view(
                padding_mask_reshaped.size(0), T, -1
            )
            padding_mask_reshaped = padding_mask_reshaped.all(-1)
            input["padding_count"] = padding_mask_reshaped.sum(-1).max().item()
            mask_indices, mask_channel_indices = self._compute_mask_indices(
                (B, T, self._C),
                padding_mask_reshaped,
            )
            input["mask_indices"] = mask_indices
            input["mask_channel_indices"] = mask_channel_indices
            out["sample_size"] = mask_indices.sum().item()

        out["net_input"] = input
        return out

    def _get_mask_indices_dims(self, size, padding=0, dilation=1):
        if size not in self._features_size_map:
            L_in = size
            for (_, kernel_size, stride) in self._conv_feature_layers:
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            self._features_size_map[size] = L_out
        return self._features_size_map[size]

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index, 1]
        return min(self.sizes[index, 1], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.profile_sort:
            order = np.array((np.arange(len(self)), self.profiles)).T
            order = np.array(sorted(order, key = lambda x: x[1]))
            return order[:,0].astype(int)
        elif self.shuffle:
            order = [np.random.permutation(len(self))] #list of 1 np.array: [np.array()]
            # return order[0]
            # NOTE: all the above is false, I am not yet understand this function :(
            # NOTE: this is not random at all
            # the list is ordered by the length of the input spectrum
            # only random in case of equal lengths
            # My current fix is to truly randomized the order
            # but in case of unbalanced dataset
            # TODO: need to add weighted sampler
            order.append(
                np.minimum(
                    np.array(self.sizes[:, 1]).squeeze(),
                    self.max_sample_size,
                )
            )
            # print("ORDER: ", order)
            # print("SORT: ", np.lexsort(order))
            # print(order[0].shape, order[1].shape)
            return np.lexsort(order)[::-1]
        else:
            return np.arange(len(self))

    def set_bucket_info(self, num_buckets):
        self.num_buckets = num_buckets
        if self.num_buckets > 0:
            self._collated_sizes = np.minimum(
                np.array(self.sizes[:, 1]).squeeze(),
                self.max_sample_size,
            )
            self.buckets = get_buckets(
                self._collated_sizes,
                self.num_buckets,
            )
            self._bucketed_sizes = get_bucketed_sizes(
                self._collated_sizes, self.buckets
            )
            logger.info(
                f"{len(self.buckets)} bucket(s) for the audio dataset: "
                f"{self.buckets}"
            )


class MelAudioDataset(STFTAudioDataset):
    def __init__(
        self,
        manifest_path,
        # sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        compute_mask_indices=False,
        text_compression_level=TextCompressionLevel.none,
        profiling=False,
        specaug=False,
        wave2graph = False,
        adresso_pretrain=False,
        profile_sort=False,
        **mask_compute_kwargs,
    ):
        super().__init__(
            # sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask_indices=compute_mask_indices,
            profiling=profiling,
            profile_sort=profile_sort,
            wave2graph=wave2graph,
            adresso_pretrain=adresso_pretrain,
            **mask_compute_kwargs,
        )

        self.text_compressor = TextCompressor(level=text_compression_level)
        self.specaug = specaug

        skipped = 0
        self.fnames = []
        sizes = []
        self.skipped_indices = set()

        self.manifest_path = manifest_path
        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                # assert len(items) == 3, line
                sz = [int(items[1]), int(items[2])]
                # if min_sample_size is not None and sz < min_sample_size:
                #     skipped += 1
                #     self.skipped_indices.add(i)
                #     continue
                self.fnames.append(self.text_compressor.compress(items[0]))
                sizes.append(sz)
                if self.profiling or self.profile_sort:
                    if len(items) == 3:
                        s = items[0]
                        profile = s[s.rfind('/')+1:-17]
                    else:
                        profile = items[3]
                    self.profiles.append(profile)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        self.sizes = np.array(sizes, dtype=np.int64)

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

        self.set_bucket_info(num_buckets)

    def __getitem__(self, index):
        import soundfile as sf

        fn = self.fnames[index]
        fn = fn if isinstance(self.fnames, list) else fn.as_py()
        fn = self.text_compressor.decompress(fn)
        path_or_fp = os.path.join(self.root_dir, fn)
        _path, slice_ptr = parse_path(path_or_fp)
        if len(slice_ptr) == 2:
            byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
            assert is_sf_audio_data(byte_data)
            path_or_fp = io.BytesIO(byte_data)

        # wav, curr_sample_rate = sf.read(path_or_fp, dtype="float32")

        # feats = torch.from_numpy(wav).float()
        # feats = self.postprocess(feats, curr_sample_rate)
        feats = torch.from_numpy(np.load(path_or_fp)).float()

        # from: vocalsound baseline code
        if self.specaug == True:
            freqm = torchaudio.transforms.FrequencyMasking(48)
            timem = torchaudio.transforms.TimeMasking(192*22.05/16)
            # feats = torch.transpose(feats, 0, 1)
            feats = feats.unsqueeze(0)
            feats = freqm(feats)
            feats = timem(feats)
            feats = feats.squeeze(0)
            # feats = torch.transpose(feats, 0, 1)
            feats = torch.roll(feats, np.random.randint(0, 1024*22.05/16), 0)

        dct = {"id": index, "source": feats}

        if self.profiling:
            profile = self.profiles[index]
            group = []
            for idx, p in enumerate(self.profiles):
                if p == profile:
                    group.append(idx)
            dct["profile"] = profile
            dct["profile_group"] = group
        if self.wave2graph:
            try:
                mfcc = torch.from_numpy(np.load(path_or_fp[:-4] + '_mfcc.npy')).float()
                corr = torch.from_numpy(np.load(path_or_fp[:-4] + '_corr.npy')).float()
            except Exception as _:
                mfcc = torch.from_numpy(np.load(path_or_fp.replace('_mel.npy', '_mfcc.npy'))).float()
                corr = torch.from_numpy(np.load(path_or_fp.replace('_mel.npy', '_corr.npy'))).float()
            if self.specaug == True:
                freqm = torchaudio.transforms.FrequencyMasking(6)
                timem = torchaudio.transforms.TimeMasking(128*22.05/16)
                # feats = torch.transpose(feats, 0, 1)
                mfcc = mfcc.unsqueeze(0)
                mfcc = freqm(mfcc)
                mfcc = timem(mfcc)
                mfcc = mfcc.squeeze(0)
                # feats = torch.transpose(feats, 0, 1)
                mfcc = torch.roll(mfcc, np.random.randint(0, 256*22.05/16), 0)
            dct['mfcc'] = mfcc
            dct['corr'] = corr
        if self.adresso_pretrain:
            # fold = self.manifest_path[:self.manifest_path.rfind('/')][-1]
            fold = self.manifest_path[self.manifest_path.find('fold')+4]
            try:
                pretrain_path = path_or_fp.replace('/audio/cn/', f'/pretrain/script/fold_{fold}/cn_').replace('_mel', '.text')
                w2v_pretrain_path = path_or_fp.replace('/audio/cn/', f'/pretrain/wav/fold_{fold}/cn_').replace('_mel', '.wav')
                os_pretrain_path = path_or_fp.replace('/audio/cn/', f'/opensmile/IS10_paraling/cn_').replace('_mel', '.IS10_paraling')
                pretrain = torch.from_numpy(np.load(pretrain_path)).float()
                w2v_pretrain = torch.from_numpy(np.load(w2v_pretrain_path)).float()
                os_pretrain = torch.from_numpy(np.load(os_pretrain_path)).float()
            except Exception as _:
                pretrain_path = path_or_fp.replace('/audio/ad/', f'/pretrain/script/fold_{fold}/ad_').replace('_mel', '.text')
                w2v_pretrain_path = path_or_fp.replace('/audio/ad/', f'/pretrain/wav/fold_{fold}/ad_').replace('_mel', '.wav')
                os_pretrain_path = path_or_fp.replace('/audio/ad/', f'/opensmile/IS10_paraling/ad_').replace('_mel', '.IS10_paraling')
                pretrain = torch.from_numpy(np.load(pretrain_path)).float()
                w2v_pretrain = torch.from_numpy(np.load(w2v_pretrain_path)).float()
                os_pretrain = torch.from_numpy(np.load(os_pretrain_path)).float()
            dct['pretrain'] = pretrain
            dct['w2v_pretrain'] = w2v_pretrain
            dct['os_pretrain'] = os_pretrain

        return dct

# class BinarizedAudioDataset(RawAudioDataset):
#     def __init__(
#         self,
#         data_dir,
#         split,
#         sample_rate,
#         max_sample_size=None,
#         min_sample_size=0,
#         shuffle=True,
#         pad=False,
#         normalize=False,
#         num_buckets=0,
#         compute_mask_indices=False,
#         **mask_compute_kwargs,
#     ):
#         super().__init__(
#             sample_rate=sample_rate,
#             max_sample_size=max_sample_size,
#             min_sample_size=min_sample_size,
#             shuffle=shuffle,
#             pad=pad,
#             normalize=normalize,
#             compute_mask_indices=compute_mask_indices,
#             **mask_compute_kwargs,
#         )

#         from fairseq.data import data_utils, Dictionary

#         self.fnames_dict = Dictionary.load(os.path.join(data_dir, "dict.txt"))

#         root_path = os.path.join(data_dir, f"{split}.root")
#         if os.path.exists(root_path):
#             with open(root_path, "r") as f:
#                 self.root_dir = next(f).strip()
#         else:
#             self.root_dir = None

#         fnames_path = os.path.join(data_dir, split)
#         self.fnames = data_utils.load_indexed_dataset(fnames_path, self.fnames_dict)
#         lengths_path = os.path.join(data_dir, f"{split}.lengths")

#         with open(lengths_path, "r") as f:
#             for line in f:
#                 sz = int(line.rstrip())
#                 assert (
#                     sz >= min_sample_size
#                 ), f"Min sample size is not supported for binarized dataset, but found a sample with size {sz}"
#                 self.sizes.append(sz)

#         self.sizes = np.array(self.sizes, dtype=np.int64)

#         self.set_bucket_info(num_buckets)
#         logger.info(f"loaded {len(self.fnames)} samples")

#     def __getitem__(self, index):
#         import soundfile as sf

#         fname = self.fnames_dict.string(self.fnames[index], separator="")
#         if self.root_dir:
#             fname = os.path.join(self.root_dir, fname)

#         wav, curr_sample_rate = sf.read(fname)
#         feats = torch.from_numpy(wav).float()
#         feats = self.postprocess(feats, curr_sample_rate)
#         return {"id": index, "source": feats}
