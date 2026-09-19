from __future__ import annotations

import io
import json
import random
import tarfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple, Any

import torch
import numpy as np
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except Exception:
    pass
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
MAX_SAME_PLACE_DISTANCE_M = 1.0


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    place_id: str
    scene_id: str
    view_id: str
    yaw_degrees: float
    split: str


def image_to_tensor(data: bytes, decoder: str = 'pil') -> torch.Tensor:
    if decoder == 'torchvision':
        try:
            from torchvision.io import ImageReadMode, decode_jpeg
            encoded = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            img = decode_jpeg(encoded, mode=ImageReadMode.RGB).float() / 255.0
            if tuple(img.shape[-2:]) != (224, 448):
                img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(224, 448), mode='bilinear', align_corners=False).squeeze(0)
            return (img.contiguous() - MEAN) / STD
        except Exception:
            # Keep the dataloader usable on systems where torchvision JPEG ops
            # are unavailable; benchmarks record which decoder is actually used.
            pass
    img = Image.open(io.BytesIO(data))
    # Preserve single-copy native ERP JPEG storage while reducing high-res decode work.
    try:
        img.draft('RGB', (448, 224))
    except Exception:
        pass
    img = img.convert('RGB').resize((448, 224))
    arr = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(224, 448, 3).float() / 255.0
    return (arr.permute(2, 0, 1).contiguous() - MEAN) / STD


def roll_tensor_by_degrees(image: torch.Tensor, yaw_degrees: float) -> Tuple[torch.Tensor, int]:
    width = int(image.shape[-1])
    shift = int(round(float(yaw_degrees) / 360.0 * width)) % width
    return torch.roll(image, shifts=shift, dims=-1), shift


def is_same_place_positive_view(
    view: Dict[str, object],
    max_distance_m: float = MAX_SAME_PLACE_DISTANCE_M,
) -> bool:
    """Return whether a rendered view may be supervised as the anchor's place."""
    distance = view.get('base_distance_m')
    if distance is None:
        return False
    try:
        value = float(distance)
    except (TypeError, ValueError):
        return False
    return 0.0 <= value <= float(max_distance_m)


def select_same_place_positive_views(
    views: Sequence[Dict[str, object]],
    count: int,
    rng: random.Random,
    shuffle: bool,
    max_distance_m: float = MAX_SAME_PLACE_DISTANCE_M,
) -> List[Dict[str, object]]:
    """Select one anchor plus nearby views, excluding boundary/far diagnostics."""
    eligible = [dict(view) for view in views if is_same_place_positive_view(view, max_distance_m)]
    anchors = [view for view in eligible if float(view.get('base_distance_m', -1.0)) == 0.0]
    non_anchors = [view for view in eligible if float(view.get('base_distance_m', -1.0)) != 0.0]
    if shuffle:
        rng.shuffle(anchors)
        rng.shuffle(non_anchors)
    ordered = anchors[:1] + non_anchors + anchors[1:]
    return ordered[: max(0, int(count))]


def split_member_name(name: str) -> Tuple[str, str]:
    for suffix in ('.jpg', '.json'):
        if name.endswith(suffix):
            return name[:-len(suffix)], suffix[1:]
    raise ValueError(f'unsupported member: {name}')


def worker_slice(items: List[Path]) -> List[Path]:
    worker = get_worker_info()
    rank = 0; world = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank(); world = torch.distributed.get_world_size()
    wid = worker.id if worker else 0
    nworkers = worker.num_workers if worker else 1
    return items[rank * nworkers + wid :: world * nworkers]


class StreamingSampleShardDataset(IterableDataset):
    """Stream sample shards without materializing a full tar into a Python dict."""
    def __init__(self, shards: Iterable[str | Path], decoder: str = 'pil'):
        self.shards = [Path(s) for s in shards]
        self.decoder = decoder

    def _place_records_by_id(self) -> Dict[str, Dict[str, object]]:
        if self._place_record_cache is None:
            self._place_record_cache = read_place_records(self.place_shards)
        return self._place_record_cache

    def _select_hard_negative_places(self, place_id: str, rng: random.Random) -> List[Dict[str, object]]:
        if self.hard_negatives_per_place <= 0 or not self.hard_negative_pairs:
            return []
        if self.hard_negative_probability < 1.0 and rng.random() > self.hard_negative_probability:
            return []
        candidates = list(self.hard_negative_pairs.get(place_id, []))
        if not candidates:
            return []
        if self.shuffle:
            rng.shuffle(candidates)
        by_id = self._place_records_by_id()
        selected: List[Dict[str, object]] = []
        for neg_id in candidates:
            row = by_id.get(str(neg_id))
            if row is None:
                continue
            copied = dict(row)
            copied['hard_negative_injected'] = True
            copied['hard_negative_for_place_id'] = place_id
            selected.append(copied)
            if len(selected) >= self.hard_negatives_per_place:
                break
        return selected

    def __iter__(self) -> Iterator[Dict[str, object]]:
        for shard in worker_slice(self.shards):
            pending: Dict[str, Dict[str, bytes]] = {}
            with tarfile.open(shard, 'r') as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    key, suffix = split_member_name(member.name)
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    group = pending.setdefault(key, {})
                    group[suffix] = f.read()
                    if 'jpg' in group and 'json' in group:
                        meta = json.loads(group['json'].decode('utf-8'))
                        yield {'image': image_to_tensor(group['jpg'], decoder=self.decoder), 'meta': meta}
                        pending.pop(key, None)


def v2_sample_shards(root: Path, split: str) -> List[Path]:
    return sorted((root / 'data/shards/samples').glob(f'{split}-samples-*.tar'))


def v2_place_shards(root: Path, split: str) -> List[Path]:
    return sorted((root / 'data/shards/places').glob(f'{split}-places-*.tar'))


def infer_split_tag_from_place_shards(place_shards: Sequence[str | Path]) -> str | None:
    tags = set()
    for shard_text in place_shards:
        name = Path(shard_text).name
        marker = '-places-'
        if marker not in name:
            return None
        tags.add(name.split(marker, 1)[0])
    return tags.pop() if len(tags) == 1 else None


def hard_negative_pairs_path_for_place_shards(place_shards: Sequence[str | Path]) -> Path | None:
    if not place_shards:
        return None
    split_tag = infer_split_tag_from_place_shards(place_shards)
    if not split_tag:
        return None
    first = Path(place_shards[0])
    data_root = first.parent.parent.parent
    return data_root / 'manifests' / f'{split_tag}_hard_negative_pairs.jsonl'


def load_hard_negative_pairs(path: str | Path | None) -> Dict[str, List[str]]:
    if path is None:
        return {}
    pair_path = Path(path)
    if not pair_path.is_file():
        return {}
    pairs: Dict[str, List[str]] = {}
    with pair_path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = row.get('query_place_id')
            n = row.get('negative_place_id')
            if q and n and q != n:
                bucket = pairs.setdefault(str(q), [])
                if str(n) not in bucket:
                    bucket.append(str(n))
    return pairs


def read_place_records(place_shards: Sequence[str | Path]) -> Dict[str, Dict[str, object]]:
    records: Dict[str, Dict[str, object]] = {}
    for shard_text in place_shards:
        shard = Path(shard_text)
        with tarfile.open(shard, 'r') as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith('.json'):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                row = json.loads(f.read().decode('utf-8'))
                place_id = row.get('place_id')
                if place_id:
                    records[str(place_id)] = row
    return records


def split_place_member_name(name: str) -> Tuple[str, str]:
    for suffix in ('.json',):
        if name.endswith(suffix):
            return name[:-len(suffix)], suffix[1:]
    raise ValueError(f'unsupported place member: {name}')


def read_sample_index(sample_shards: Sequence[str | Path]) -> Dict[str, Tuple[Path, str]]:
    """Return sample_id -> (shard, tar jpg member name), reading metadata only."""
    cached = read_cached_sample_index(sample_shards)
    if cached is not None:
        return cached
    index: Dict[str, Tuple[Path, str]] = {}
    for shard_text in sample_shards:
        shard = Path(shard_text)
        pending: Dict[str, Dict[str, str]] = {}
        with tarfile.open(shard, 'r') as tar:
            for member in tar:
                if not member.isfile():
                    continue
                key, suffix = split_member_name(member.name)
                if suffix == 'jpg':
                    pending.setdefault(key, {})['jpg_member'] = member.name
                elif suffix == 'json':
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    meta = json.loads(f.read().decode('utf-8'))
                    sample_id = meta['sample_id']
                    group = pending.setdefault(key, {})
                    group['sample_id'] = sample_id
                group = pending.get(key, {})
                if 'sample_id' in group and 'jpg_member' in group:
                    index[group['sample_id']] = (shard, group['jpg_member'])
                    pending.pop(key, None)
    return index


def infer_split_tag_from_sample_shards(sample_shards: Sequence[str | Path]) -> str | None:
    tags = set()
    for shard_text in sample_shards:
        name = Path(shard_text).name
        marker = '-samples-'
        if marker not in name:
            return None
        tags.add(name.split(marker, 1)[0])
    return tags.pop() if len(tags) == 1 else None


def sample_index_path_for_shards(sample_shards: Sequence[str | Path]) -> Path | None:
    if not sample_shards:
        return None
    split_tag = infer_split_tag_from_sample_shards(sample_shards)
    if not split_tag:
        return None
    first = Path(sample_shards[0])
    # .../v2/data/shards/samples/<tag>-samples-000000.tar -> .../v2/data/manifests
    data_root = first.parent.parent.parent
    return data_root / 'manifests' / f'{split_tag}_sample_index.json'


def read_cached_sample_index(sample_shards: Sequence[str | Path]) -> Dict[str, Tuple[Path, str]] | None:
    index_path = sample_index_path_for_shards(sample_shards)
    if index_path is None or not index_path.is_file():
        return None
    payload = json.loads(index_path.read_text(encoding='utf-8'))
    # Manifests are frozen with the builder host's absolute paths. Resolve each
    # record against the shards supplied by this runtime so the same dataset can
    # be moved without rewriting or weakening the frozen manifest.
    shards_by_name = {Path(s).name: Path(s) for s in sample_shards}
    index: Dict[str, Tuple[Path, str]] = {}
    for sample_id, row in payload.get('samples', {}).items():
        shard = shards_by_name.get(Path(row['shard']).name)
        if shard is not None:
            index[sample_id] = (shard, row['jpg_member'])
    if not index:
        raise RuntimeError(f'cached sample index has no matching shards: {index_path}')
    return index


class TarImageStore:
    def __init__(self, sample_index: Dict[str, Tuple[Path, str]], max_open_shards: int = 64):
        self.sample_index = sample_index
        self.max_open_shards = max(1, int(max_open_shards))
        self._open: "OrderedDict[Path, tarfile.TarFile]" = OrderedDict()

    def close(self) -> None:
        for tar in list(self._open.values()):
            tar.close()
        self._open.clear()

    def _get_tar(self, shard: Path) -> tarfile.TarFile:
        tar = self._open.get(shard)
        if tar is not None:
            self._open.move_to_end(shard)
            return tar
        while len(self._open) >= self.max_open_shards:
            _, old_tar = self._open.popitem(last=False)
            old_tar.close()
        tar = tarfile.open(shard, 'r')
        self._open[shard] = tar
        return tar

    def read_jpeg(self, sample_id: str) -> bytes:
        shard, member_name = self.sample_index[sample_id]
        tar = self._get_tar(shard)
        f = tar.extractfile(member_name)
        if f is None:
            raise KeyError(f'missing {member_name} in {shard}')
        return f.read()


class DecodedImageStore:
    """Read fixed-size uint8 ERP tensors from a shared read-only memmap."""

    def __init__(
        self,
        manifest_path: str | Path,
        sample_to_row: Dict[str, int] | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        self.shape = tuple(int(value) for value in manifest['shape'])
        if self.shape[1:] != (3, 224, 448):
            raise ValueError(f'unsupported decoded cache shape: {self.shape}')
        self.cache_path = Path(manifest['cache_path'])
        self.index_path = Path(manifest['index_path'])
        self.sample_to_row = sample_to_row if sample_to_row is not None else {
            str(sample_id): int(row)
            for sample_id, row in json.loads(self.index_path.read_text(encoding='utf-8')).items()
        }
        self._mmap: np.memmap | None = None

    def _array(self) -> np.memmap:
        if self._mmap is None:
            # Copy-on-write mode makes torch.from_numpy safe without allowing
            # workers to alter the shared cache file.
            self._mmap = np.memmap(
                self.cache_path,
                dtype=np.uint8,
                mode='c',
                shape=self.shape,
            )
        return self._mmap

    def read_tensor(self, sample_id: str) -> torch.Tensor:
        row = self.sample_to_row.get(str(sample_id))
        if row is None:
            raise KeyError(f'sample is absent from decoded cache: {sample_id}')
        return torch.from_numpy(self._array()[row])

    def close(self) -> None:
        self._mmap = None


class PlaceShardDataset(IterableDataset):
    """Yield P places x V views batches without pair-duplicated JPEG storage."""

    def __init__(
        self,
        place_shards: Iterable[str | Path],
        sample_shards: Iterable[str | Path],
        places_per_batch: int = 8,
        views_per_place: int = 4,
        decoder: str = 'pil',
        shuffle: bool = True,
        seed: int = 20260730,
        include_synthetic_roll: bool = True,
        synthetic_roll_yaws: Sequence[float] = (15.0, 30.0, 60.0, 90.0),
        hard_negative_pairs_path: str | Path | None = None,
        hard_negatives_per_place: int = 1,
        hard_negative_probability: float = 1.0,
        max_same_place_distance_m: float = MAX_SAME_PLACE_DISTANCE_M,
        rank: int = 0,
        world_size: int = 1,
        preload_metadata: bool = True,
        decoded_cache_manifest: str | Path | None = None,
    ):
        self.place_shards = [Path(s) for s in place_shards]
        self.sample_shards = [Path(s) for s in sample_shards]
        self.places_per_batch = int(places_per_batch)
        self.views_per_place = int(views_per_place)
        self.decoder = decoder
        self.shuffle = shuffle
        self.seed = int(seed)
        self.include_synthetic_roll = bool(include_synthetic_roll)
        self.synthetic_roll_yaws = [float(x) for x in synthetic_roll_yaws]
        inferred_hard_path = hard_negative_pairs_path_for_place_shards(self.place_shards)
        self.hard_negative_pairs_path = Path(hard_negative_pairs_path) if hard_negative_pairs_path else inferred_hard_path
        self.hard_negative_pairs = load_hard_negative_pairs(self.hard_negative_pairs_path)
        self.hard_negatives_per_place = max(0, int(hard_negatives_per_place))
        self.hard_negative_probability = max(0.0, min(1.0, float(hard_negative_probability)))
        self.max_same_place_distance_m = float(max_same_place_distance_m)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.decoded_cache_manifest = Path(decoded_cache_manifest) if decoded_cache_manifest else None
        self._decoded_cache_index: Dict[str, int] | None = None
        self._place_record_cache: Dict[str, Dict[str, object]] | None = None
        self._gallery_records: List[Dict[str, object]] | None = None
        self._sample_index: Dict[str, Tuple[Path, str]] | None = None
        if preload_metadata:
            self._place_record_cache = read_place_records(self.place_shards)
            self._gallery_records = [
                row for row in self._place_record_cache.values()
                if row.get('role') == 'gallery_place'
            ]
            if self.decoded_cache_manifest:
                decoded_manifest = json.loads(self.decoded_cache_manifest.read_text(encoding='utf-8'))
                self._decoded_cache_index = {
                    str(sample_id): int(row)
                    for sample_id, row in json.loads(
                        Path(decoded_manifest['index_path']).read_text(encoding='utf-8')
                    ).items()
                }
            else:
                self._sample_index = read_sample_index(self.sample_shards)

    def _iter_place_records(self, rng: random.Random) -> Iterator[Dict[str, object]]:
        worker = get_worker_info()
        wid = worker.id if worker else 0
        nworkers = worker.num_workers if worker else 1
        stride = max(1, self.world_size * int(nworkers))
        offset = self.rank * int(nworkers) + int(wid)
        order_rng = random.Random(self.seed)
        records = self._gallery_records
        if records is None:
            records = [
                row for row in read_place_records(self.place_shards).values()
                if row.get('role') == 'gallery_place'
            ]
        records = list(records)
        if self.shuffle:
            order_rng.shuffle(records)
        yield from records[offset::stride]

    def _place_records_by_id(self) -> Dict[str, Dict[str, object]]:
        if self._place_record_cache is None:
            self._place_record_cache = read_place_records(self.place_shards)
        return self._place_record_cache

    def _select_hard_negative_places(self, place_id: str, rng: random.Random) -> List[Dict[str, object]]:
        if self.hard_negatives_per_place <= 0 or not self.hard_negative_pairs:
            return []
        if self.hard_negative_probability < 1.0 and rng.random() > self.hard_negative_probability:
            return []
        candidates = list(self.hard_negative_pairs.get(place_id, []))
        if not candidates:
            return []
        if self.shuffle:
            rng.shuffle(candidates)
        by_id = self._place_records_by_id()
        selected: List[Dict[str, object]] = []
        for neg_id in candidates:
            row = by_id.get(str(neg_id))
            if row is None:
                continue
            copied = dict(row)
            copied['hard_negative_injected'] = True
            copied['hard_negative_for_place_id'] = place_id
            selected.append(copied)
            if len(selected) >= self.hard_negatives_per_place:
                break
        return selected

    def __iter__(self) -> Iterator[Dict[str, object]]:
        worker = get_worker_info()
        wid = worker.id if worker else 0
        rng = random.Random(self.seed + self.rank * 1009 + wid)
        sample_index = None
        if not self.decoded_cache_manifest:
            sample_index = self._sample_index or read_sample_index(self.sample_shards)
        store = (
            DecodedImageStore(self.decoded_cache_manifest, self._decoded_cache_index)
            if self.decoded_cache_manifest
            else TarImageStore(sample_index)
        )
        batch_places: List[Dict[str, object]] = []
        try:
            for place in self._iter_place_records(rng):
                views = select_same_place_positive_views(
                    list(place.get('views', [])),
                    self.views_per_place,
                    rng,
                    self.shuffle,
                    self.max_same_place_distance_m,
                )
                if len(views) < self.views_per_place:
                    continue
                place = dict(place)
                place['views'] = views
                batch_places.append(place)
                for hard_place in self._select_hard_negative_places(str(place['place_id']), rng):
                    hard_views = select_same_place_positive_views(
                        list(hard_place.get('views', [])),
                        self.views_per_place,
                        rng,
                        self.shuffle,
                        self.max_same_place_distance_m,
                    )
                    if len(hard_views) < self.views_per_place:
                        continue
                    hard_place = dict(hard_place)
                    hard_place['views'] = hard_views
                    batch_places.append(hard_place)
                if len(batch_places) >= self.places_per_batch:
                    yield self._make_batch(batch_places[:self.places_per_batch], store)
                    batch_places = batch_places[self.places_per_batch:]
            # Do not emit a partial worker batch. The outer assembler combines
            # fixed-size place micro-batches; a short tail can make a nominally
            # divisible target overshoot at epoch boundaries.
        finally:
            store.close()

    def _make_batch(self, places: List[Dict[str, object]], store: TarImageStore) -> Dict[str, object]:
        images = []
        metas = []
        place_ids = []
        view_ids = []
        precise_yaw_mask = []
        yaw_degrees = []
        for place in places:
            pid = str(place['place_id'])
            for view in place['views']:
                sample_id = str(view['sample_id'])
                if isinstance(store, DecodedImageStore):
                    image = store.read_tensor(sample_id)
                else:
                    image = image_to_tensor(store.read_jpeg(sample_id), decoder=self.decoder)
                meta = dict(view)
                meta['place_id'] = pid
                meta['scene_id'] = place.get('scene_id')
                meta['synthetic_roll_applied'] = False
                meta['hard_negative_injected'] = bool(place.get('hard_negative_injected', False))
                if place.get('hard_negative_for_place_id') is not None:
                    meta['hard_negative_for_place_id'] = str(place.get('hard_negative_for_place_id'))
                meta['synthetic_roll_shift_pixels'] = 0
                metas.append(meta)
                images.append(image)
                place_ids.append(pid)
                view_ids.append(sample_id)
                precise_yaw_mask.append(bool(view.get('precise_yaw_label', False)))
                yaw_degrees.append(float(view.get('yaw_degrees', 0.0)))
                if self.include_synthetic_roll and view.get('synthetic_roll_yaws_available'):
                    for roll_yaw in self.synthetic_roll_yaws:
                        rolled, shift = roll_tensor_by_degrees(image, roll_yaw)
                        roll_meta = dict(meta)
                        roll_meta['sample_id'] = f'{sample_id}__roll{int(round(roll_yaw)) % 360:03d}'
                        roll_meta['synthetic_roll_applied'] = True
                        roll_meta['synthetic_roll_shift_pixels'] = int(shift)
                        roll_meta['yaw_degrees'] = float(roll_yaw)
                        roll_meta['precise_yaw_label'] = True
                        metas.append(roll_meta)
                        images.append(rolled)
                        place_ids.append(pid)
                        view_ids.append(roll_meta['sample_id'])
                        precise_yaw_mask.append(True)
                        yaw_degrees.append(float(roll_yaw))
        return {
            'image': torch.stack(images),
            'meta': metas,
            'place_ids': place_ids,
            'view_ids': view_ids,
            'precise_yaw_mask': torch.tensor(precise_yaw_mask, dtype=torch.bool),
            'yaw_degrees': torch.tensor(yaw_degrees, dtype=torch.float32),
            'places_in_batch': len(places),
            'views_per_place': self.views_per_place,
            'hard_negative_pairs_path': str(self.hard_negative_pairs_path) if self.hard_negative_pairs_path else None,
            'hard_negative_pair_count': sum(len(v) for v in self.hard_negative_pairs.values()),
            'hard_negative_place_count': sum(1 for p in places if p.get('hard_negative_injected')),
        }


def merge_place_batches(batches: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Merge worker-produced place micro-batches without changing supervision."""
    if not batches:
        raise ValueError('at least one place micro-batch is required')
    first = batches[0]
    image = torch.cat([batch['image'] for batch in batches], dim=0)
    precise_yaw_mask = torch.cat([batch['precise_yaw_mask'] for batch in batches], dim=0)
    yaw_degrees = torch.cat([batch['yaw_degrees'] for batch in batches], dim=0)
    if torch.cuda.is_available():
        image = image.pin_memory()
        precise_yaw_mask = precise_yaw_mask.pin_memory()
        yaw_degrees = yaw_degrees.pin_memory()
    return {
        'image': image,
        'meta': [item for batch in batches for item in batch['meta']],
        'place_ids': [item for batch in batches for item in batch['place_ids']],
        'view_ids': [item for batch in batches for item in batch['view_ids']],
        'precise_yaw_mask': precise_yaw_mask,
        'yaw_degrees': yaw_degrees,
        'places_in_batch': sum(int(batch['places_in_batch']) for batch in batches),
        'views_per_place': int(first['views_per_place']),
        'hard_negative_pairs_path': first.get('hard_negative_pairs_path'),
        'hard_negative_pair_count': int(first.get('hard_negative_pair_count', 0)),
        'hard_negative_place_count': sum(int(batch.get('hard_negative_place_count', 0)) for batch in batches),
        'loader_micro_batch_count': len(batches),
    }


class AssembledPlaceBatchLoader:
    """Assemble parallel DataLoader micro-batches into one optimizer-step batch."""

    def __init__(self, loader: object, target_places: int):
        self.loader = loader
        self.target_places = int(target_places)

    def __iter__(self) -> Iterator[Dict[str, object]]:
        pending: List[Dict[str, object]] = []
        places = 0
        for batch in self.loader:
            pending.append(batch)
            places += int(batch['places_in_batch'])
            if places == self.target_places:
                yield merge_place_batches(pending)
                pending = []
                places = 0
            elif places > self.target_places:
                raise RuntimeError(
                    f'place micro-batches overshot target: {places} > {self.target_places}'
                )
