import contextlib
import glob
import hashlib
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torchvision
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed

try:
    from ultralytics.data.build import seed_worker
except Exception:
    def seed_worker(worker_id):
        seed = torch.initial_seed() % 2**32
        random.seed(seed)
        np.random.seed(seed)

try:
    from ultralytics.data.utils import get_hash, img2label_paths
except Exception:
    def get_hash(paths):
        size = sum(os.path.getsize(x) for x in paths if os.path.exists(x))
        return hashlib.sha256("".join(paths).encode()).hexdigest() if size else ""

    def img2label_paths(paths):
        a, b = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
        return [b.join(x.rsplit(a, 1)).rsplit(".", 1)[0] + ".txt" for x in paths]

from utils.augmentations import (
    Albumentations,
    augment_hsv,
    classify_albumentations,
    classify_transforms,
    copy_paste,
    letterbox,
    mixup,
    random_perspective,
)
from utils.general import (
    LOGGER,
    NUM_THREADS,
    TQDM,
    check_requirements,
    clean_str,
    cv2,
    is_colab,
    is_kaggle,
    segments2boxes,
    xyn2xy,
    xywhn2xyxy,
    xyxy2xywhn,
)
from utils.torch_utils import torch_distributed_zero_first

HELP_URL = "See https://docs.ultralytics.com/yolov5/tutorials/train-custom-data"
IMG_FORMATS = ("bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm")
VID_FORMATS = ("asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv")
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "-1"))
RANK = int(os.getenv("RANK", "-1"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
PIN_MEMORY = str(os.getenv("PIN_MEMORY", "true")).lower() == "true"
orientation = next((k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None)


def exif_size(img):
    s = img.size
    with contextlib.suppress(Exception):
        if dict(img._getexif().items()).get(orientation) in (6, 8):
            s = s[1], s[0]
    return s


class SmartDistributedSampler(distributed.DistributedSampler):
    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        n = int((len(self.dataset) - self.rank - 1) / self.num_replicas) + 1
        indices = torch.randperm(n, generator=generator)
        if not self.shuffle:
            indices = indices.sort()[0]
        indices = indices.tolist()
        if self.drop_last:
            indices = indices[:self.num_samples]
        else:
            remaining = self.num_samples - len(indices)
            if remaining <= len(indices):
                indices += indices[:remaining]
            else:
                indices += (indices * math.ceil(remaining / len(indices)))[:remaining]
        return iter(indices)


def create_dataloader(
    path,
    imgsz,
    batch_size,
    stride,
    single_cls=False,
    hyp=None,
    augment=False,
    cache=False,
    pad=0.0,
    rect=False,
    rank=-1,
    workers=8,
    image_weights=False,
    quad=False,
    prefix="",
    shuffle=False,
    seed=0,
):
    if rect and shuffle:
        LOGGER.warning("--rect is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,
            hyp=hyp,
            rect=rect,
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            rank=rank,
        )
    batch_size = min(batch_size, len(dataset))
    device_count = torch.cuda.device_count()
    workers = min(os.cpu_count() // max(device_count, 1), batch_size if batch_size > 1 else 0, workers)
    sampler = None if rank == -1 else SmartDistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    return (
        loader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            num_workers=workers,
            sampler=sampler,
            drop_last=quad,
            pin_memory=PIN_MEMORY,
            collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn,
            worker_init_fn=seed_worker,
            generator=generator,
        ),
        dataset,
    )


class InfiniteDataLoader(dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadScreenshots:
    def __init__(self, source, img_size=640, stride=32, auto=True, transforms=None):
        check_requirements("mss")
        import mss

        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = map(int, params)
        elif len(params) == 5:
            self.screen, left, top, width, height = map(int, params)
        self.img_size, self.stride, self.transforms, self.auto = img_size, stride, transforms, auto
        self.mode, self.frame = "stream", 0
        self.sct = mss.mss()
        monitor = self.sct.monitors[self.screen]
        self.top = monitor["top"] if top is None else monitor["top"] + top
        self.left = monitor["left"] if left is None else monitor["left"] + left
        self.width = width or monitor["width"]
        self.height = height or monitor["height"]
        self.monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def __iter__(self):
        return self

    def __next__(self):
        im0 = np.asarray(self.sct.grab(self.monitor))[:, :, :3]
        status = f"screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: "
        if self.transforms:
            im = self.transforms(im0)
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]
            im = im.transpose((2, 0, 1))[::-1]
            im = np.ascontiguousarray(im)
        self.frame += 1
        return str(self.screen), im, im0, None, status


class LoadImages:
    def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        if isinstance(path, (str, Path)):
            path = str(Path(path).resolve())
            if "*" in path:
                files = sorted(glob.glob(path, recursive=True))
            elif os.path.isdir(path):
                files = sorted(glob.glob(os.path.join(path, "*.*")))
            elif os.path.isfile(path):
                files = [path]
            else:
                raise FileNotFoundError(f"{path} does not exist")
        else:
            files = [str(x) for x in path]
        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split(".")[-1].lower() in VID_FORMATS]
        self.files = images + videos
        self.nf = len(self.files)
        self.video_flag = [False] * len(images) + [True] * len(videos)
        self.mode = "image"
        self.auto = auto
        self.transforms = transforms
        self.img_size = img_size
        self.stride = stride
        self.vid_stride = vid_stride
        self.cap = None
        if videos:
            self._new_video(videos[0])
        else:
            self.cap = None
        assert self.nf, f"No images or videos found in {path}. Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        if self.video_flag[self.count]:
            self.mode = "video"
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()
            self.frame += 1
            status = f"video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: "
        else:
            self.count += 1
            im0 = cv2.imread(path)
            assert im0 is not None, f"Image Not Found {path}"
            status = f"image {self.count}/{self.nf} {path}: "
        if self.transforms:
            im = self.transforms(im0)
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]
            im = im.transpose((2, 0, 1))[::-1]
            im = np.ascontiguousarray(im)
        return path, im, im0, self.cap, status

    def _new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))
        if cv2.__version__ >= "4.5.4":
            self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    def __len__(self):
        return self.nf


class LoadStreams:
    def __init__(self, sources="streams.txt", img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        self.mode = "stream"
        self.img_size, self.stride, self.auto = img_size, stride, auto
        self.transforms, self.vid_stride = transforms, vid_stride
        if os.path.isfile(sources):
            sources = Path(sources).read_text().rsplit()
        else:
            sources = [sources]
        self.sources = [clean_str(x) for x in sources]
        self.imgs, self.fps, self.frames, self.threads, self.caps = [], [], [], [], []
        self.running = True
        for i, source in enumerate(sources):
            source = eval(source) if source.isnumeric() else source
            if source == "youtube":
                check_requirements(("pafy", "youtube_dl==2020.12.2"))
                source = pafy.new(source).getbest(preftype="mp4").url
            cap = cv2.VideoCapture(source)
            assert cap.isOpened(), f"{source} failed to open"
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.frames.append(max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0))
            self.fps.append(fps if math.isfinite(fps) and fps > 0 else 30)
            success, image = cap.read()
            assert success, f"{source} failed to read"
            self.imgs.append(image)
            self.caps.append(cap)
            thread = Thread(target=self.update, args=(i, cap, source), daemon=True)
            thread.start()
            self.threads.append(thread)
            LOGGER.info(f"{i + 1}/{len(sources)}: {source}... Success ({self.frames[-1]} frames {width}x{height} at {self.fps[-1]:.2f} FPS)")
        shapes = np.stack([letterbox(x, img_size, stride=stride, auto=auto)[0].shape for x in self.imgs])
        self.rect = np.unique(shapes, axis=0).shape[0] == 1
        self.auto = auto and self.rect
        self.count = -1
        if not self.rect:
            LOGGER.warning("WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.")

    def update(self, i, cap, stream):
        n, f = 0, self.frames[i]
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()
            if n % self.vid_stride == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning(f"WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.")
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)
            time.sleep(0.0)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            raise StopIteration
        im0 = self.imgs.copy()
        if self.transforms:
            im = np.stack([self.transforms(x) for x in im0])
        else:
            im = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in im0])
            im = im[:, :, :, ::-1].transpose((0, 3, 1, 2))
            im = np.ascontiguousarray(im)
        return self.sources, im, im0, None, ""

    def __len__(self):
        return len(self.sources)


def img2label_paths_local(img_paths):
    return img2label_paths(img_paths)


class LoadImagesAndLabels(Dataset):
    cache_version = 0.6
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        min_items=0,
        prefix="",
        rank=-1,
    ):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp or {}
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations(size=img_size) if augment else None
        try:
            files = []
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)
                if p.is_dir():
                    files.extend(glob.glob(str(p / "**" / "*.*"), recursive=True))
                elif p.is_file():
                    with open(p) as f:
                        entries = f.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    files.extend([x.replace("./", parent, 1) if x.startswith("./") else x for x in entries])
                else:
                    raise FileNotFoundError(f"{prefix}{p} does not exist")
            self.im_files = sorted(x.replace("/", os.sep) for x in files if x.split(".")[-1].lower() in IMG_FORMATS)
            assert self.im_files, f"{prefix}No images found"
        except Exception as e:
            raise Exception(f"{prefix}Error loading data from {path}\n{HELP_URL}") from e
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True
            assert cache["version"] == self.cache_version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False
        nf, nm, ne, nc, n = cache.pop("results")
        if exists and LOCAL_RANK in (-1, 0):
            d = f"{prefix}Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            for _ in TQDM(None, desc=d, total=n, initial=n, bar_format="{l_bar}{bar:10}{r_bar}"):
                pass
        assert nf > 0 or not augment, f"{prefix}No labels found in {cache_path}, can not start training. {HELP_URL}"
        cache.pop("hash", None)
        cache.pop("version", None)
        self.im_files = list(cache.keys())
        self.labels = [cache[x][0] for x in self.im_files]
        self.segments = [cache[x][1] for x in self.im_files]
        self.shapes = np.array([cache[x][2] for x in self.im_files], dtype=np.float64)
        self.label_files = img2label_paths(self.im_files)
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0
        if min_items:
            include = np.array([len(x) >= min_items for x in self.labels])
            self.im_files = [x for i, x in enumerate(self.im_files) if include[i]]
            self.label_files = [x for i, x in enumerate(self.label_files) if include[i]]
            self.labels = [x for i, x in enumerate(self.labels) if include[i]]
            self.segments = [x for i, x in enumerate(self.segments) if include[i]]
            self.shapes = self.shapes[include]
        self.n = len(self.shapes)
        self.indices = range(self.n)
        self.batch = np.floor(np.arange(self.n) / batch_size).astype(int)
        nb = self.batch[-1] + 1
        if self.rect:
            aspect = self.shapes[:, 1] / self.shapes[:, 0]
            order = aspect.argsort()
            self.im_files = [self.im_files[i] for i in order]
            self.label_files = [self.label_files[i] for i in order]
            self.labels = [self.labels[i] for i in order]
            self.segments = [self.segments[i] for i in order]
            self.shapes = self.shapes[order]
            aspect = aspect[order]
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ar = aspect[self.batch == i]
                mini, maxi = ar.min(), ar.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride
        self.batch = self.batch
        self.imgs = [None] * self.n
        self.ims = self.imgs
        self.im_hw0 = [None] * self.n
        self.im_hw = [None] * self.n
        self.npy_files = [Path(x).with_suffix(".npy") for x in self.im_files]
        if cache_images:
            gb = 0
            if cache_images == "ram":
                results = ThreadPool(NUM_THREADS).imap(lambda x: self.load_image(*x), zip(repeat(self), range(self.n)))
                pbar = TQDM(enumerate(results), total=self.n, bar_format="{l_bar}{bar:10}{r_bar}", desc=f"{prefix}Caching images")
                for i, x in pbar:
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x
                    gb += self.ims[i].nbytes
                    pbar.desc = f"{prefix}Caching images ({gb / 1e9:.1f}GB RAM)"
            elif cache_images == "disk":
                pbar = TQDM(self.npy_files, total=self.n, desc=f"{prefix}Caching images")
                for i, x in enumerate(pbar):
                    if not x.exists():
                        np.save(x.as_posix(), cv2.imread(self.im_files[i]))
                    gb += x.stat().st_size
                    pbar.desc = f"{prefix}Caching images ({gb / 1e9:.1f}GB disk)"

    def cache_labels(self, path=Path("./labels.cache"), prefix=""):
        x = {}
        nm, nf, ne, nc = 0, 0, 0, 0
        messages = []
        desc = f"{prefix}Scanning {path.parent / path.stem}..."
        with Pool(NUM_THREADS) as pool:
            args = zip(self.im_files, self.label_files, repeat(prefix))
            pbar = TQDM(pool.imap(verify_image_label, args), desc=desc, total=len(self.im_files), bar_format="{l_bar}{bar:10}{r_bar}")
            for im_file, labels, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [labels, segments, shape]
                if msg:
                    messages.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
        pbar.close()
        if messages:
            LOGGER.info("\n".join(messages))
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["messages"] = messages
        x["version"] = self.cache_version
        if nf:
            path.parent.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(Exception):
                np.save(str(path), x)
                Path(str(path) + ".npy").rename(path)
                LOGGER.info(f"{prefix}New cache created: {path}")
        return x

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, index):
        index = self.indices[index]
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp.get("mosaic", 0.0)
        if mosaic:
            img, labels = self.load_mosaic(index)
            shapes = None
            if random.random() < hyp.get("mixup", 0.0):
                img2, labels2 = self.load_mosaic(random.choice(self.indices))
                img, labels = mixup(img, labels, img2, labels2)
        else:
            img, (h0, w0), (h, w) = self.load_image(index)
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)
            labels = self.labels[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
            if self.augment:
                img, labels = random_perspective(
                    img,
                    labels,
                    segments=self.segments[index].copy(),
                    degrees=hyp.get("degrees", 0.0),
                    translate=hyp.get("translate", 0.0),
                    scale=hyp.get("scale", 0.0),
                    shear=hyp.get("shear", 0.0),
                    perspective=hyp.get("perspective", 0.0),
                )
        nl = len(labels)
        if self.augment:
            if not mosaic:
                img, labels = copy_paste(img, labels, self.segments[index], p=hyp.get("copy_paste", 0.0))
            if self.albumentations:
                img, labels = self.albumentations(img, labels)
            nl = len(labels)
            augment_hsv(img, hgain=hyp.get("hsv_h", 0.0), sgain=hyp.get("hsv_s", 0.0), vgain=hyp.get("hsv_v", 0.0))
            if random.random() < hyp.get("flipud", 0.0):
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]
            if random.random() < hyp.get("fliplr", 0.0):
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    def load_image(self, i):
        im, f = self.ims[i], self.npy_files[i]
        if im is None:
            if f.exists():
                im = np.load(f)
            else:
                im = cv2.imread(self.im_files[i])
            assert im is not None, f"Image Not Found {self.im_files[i]}"
            h0, w0 = im.shape[:2]
            r = self.img_size / max(h0, w0)
            if r != 1:
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=self.rand_interp_methods[random.randint(0, 4)] if self.augment else cv2.INTER_LINEAR)
            return im, (h0, w0), im.shape[:2]
        return im, self.im_hw0[i], self.im_hw[i]

    def load_mosaic(self, index):
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)
        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)
        for i, index in enumerate(indices):
            img, _, (h, w) = self.load_image(index)
            if i == 0:
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw, padh = x1a - x1b, y1a - y1b
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)
        img4, labels4 = copy_paste(img4, labels4, segments4, p=self.hyp.get("copy_paste", 0.0))
        img4, labels4 = random_perspective(
            img4,
            labels4,
            segments4,
            degrees=self.hyp.get("degrees", 0.0),
            translate=self.hyp.get("translate", 0.0),
            scale=self.hyp.get("scale", 0.0),
            shear=self.hyp.get("shear", 0.0),
            perspective=self.hyp.get("perspective", 0.0),
            border=self.mosaic_border,
        )
        return img4, labels4

    @staticmethod
    def collate_fn(batch):
        im, label, path, shapes = zip(*batch)
        for i, lb in enumerate(label):
            lb[:, 0] = i
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        im, label, path, shapes = zip(*batch)
        n = len(shapes) // 4
        im4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]
        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])
        for i in range(n):
            i *= 4
            if random.random() < 0.5:
                image = F.interpolate(im[i].unsqueeze(0).float(), scale_factor=2.0, mode="bilinear", align_corners=False)[0].type(im[i].type())
                labels = label[i]
            else:
                image = torch.cat((torch.cat((im[i], im[i + 1]), 1), torch.cat((im[i + 2], im[i + 3]), 1)), 2)
                labels = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            im4.append(image)
            label4.append(labels)
        for i, lb in enumerate(label4):
            lb[:, 0] = i
        return torch.stack(im4, 0), torch.cat(label4, 0), path4, shapes4


class ClassificationDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, augment, img_size, cache=False):
        super().__init__(root=root)
        self.cache_ram = cache == "ram"
        self.cache_disk = cache == "disk"
        self.samples = self.samples
        self.album_transform = classify_albumentations(augment, img_size) if augment else None
        self.torch_transforms = classify_transforms(img_size)
        self.npy_files = [Path(x[0]).with_suffix(".npy") for x in self.samples]
        self.ims = [None] * len(self.samples)
        self.im_hw0, self.im_hw = [None] * len(self.samples), [None] * len(self.samples)

    def __getitem__(self, index):
        path, class_index = self.samples[index]
        im = self.ims[index]
        if im is None:
            npy = self.npy_files[index]
            if self.cache_disk and npy.exists():
                im = np.load(npy)
            else:
                im = cv2.imread(path)
                if self.cache_disk:
                    np.save(npy, im)
                elif self.cache_ram:
                    self.ims[index] = im
        if self.album_transform:
            im = self.album_transform(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))["image"]
        else:
            im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        return self.torch_transforms(im), class_index


def create_classification_dataloader(path, imgsz=224, batch_size=16, augment=False, cache=False, rank=-1, workers=8, shuffle=True):
    with torch_distributed_zero_first(rank):
        dataset = ClassificationDataset(path, augment, imgsz, cache)
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()
    nw = min(os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers)
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def verify_image_label(args):
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, "", []
    try:
        im = Image.open(im_file)
        im.verify()
        shape = exif_size(Image.open(im_file))
        assert shape[0] > 9 and shape[1] > 9, f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"
        if os.path.isfile(lb_file):
            nf = 1
            with open(lb_file) as f:
                lines = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lines):
                    classes = np.array([x[0] for x in lines], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lines]
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
                else:
                    lb = np.array(lines, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
                assert (lb[:, 1:] <= 1).all(), f"non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}"
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:
                    lb = lb[i]
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return None, None, None, None, nm, nf, ne, nc, msg


def get_cache(path):
    cache = {}
    if path.exists():
        with contextlib.suppress(Exception):
            cache = np.load(path, allow_pickle=True).item()
    return cache


def img2label_paths2(img_paths):
    return img2label_paths(img_paths)


def flatten_recursive(path=Path("../datasets/coco128")):
    path = Path(path)
    new_path = Path(str(path) + "_flat")
    new_path.mkdir(parents=True, exist_ok=True)
    for file in TQDM(path.rglob("*.*"), desc=f"Flattening {path}"):
        if file.is_file():
            shutil.copyfile(file, new_path / file.name)


def extract_boxes(path=Path("../datasets/coco128")):
    path = Path(path)
    files = list(path.rglob("*.*"))
    images = [x for x in files if x.suffix[1:].lower() in IMG_FORMATS]
    for im_file in TQDM(images, desc=f"Extracting boxes from {path}"):
        label_file = Path(img2label_paths([str(im_file)])[0])
        if label_file.exists():
            im = cv2.imread(str(im_file))
            h, w = im.shape[:2]
            labels = np.loadtxt(label_file, ndmin=2)
            for j, x in enumerate(labels):
                class_id = int(x[0])
                x = xywhn2xyxy(x[1:].reshape(-1, 4), w, h)[0].astype(int)
                crop = im[x[1]:x[3], x[0]:x[2]]
                if crop.size:
                    out = path / "classifier" / str(class_id) / f"{im_file.stem}_{j}.jpg"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out), crop)


def autosplit(path=Path("../datasets/coco128/images"), weights=(0.9, 0.1, 0.0), annotated_only=False):
    path = Path(path)
    files = sum([list(path.rglob(f"*.{x}")) for x in IMG_FORMATS], [])
    n = len(files)
    indices = random.choices([0, 1, 2], weights=weights, k=n)
    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"]
    LOGGER.info(f"Autosplitting images from {path}")
    for x in txt:
        (path.parent / x).unlink(missing_ok=True)
    for i, im in TQDM(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(im)])[0]).exists():
            with open(path.parent / txt[i], "a") as f:
                f.write("./" + im.relative_to(path.parent).as_posix() + "\n")


class HUBDatasetStats:
    def __init__(self, path="coco128.yaml", task="detect", autodownload=False):
        from utils.general import check_dataset, yaml_load

        self.task = task
        self.data = yaml_load(path)
        if self.data.get("path"):
            self.data["path"] = str(Path(self.data["path"]).resolve())
        self.data = check_dataset(self.data, autodownload)
        self.hub_dir = Path(str(self.data["path"]) + "-hub")
        self.stats = {"nc": self.data["nc"], "names": list(self.data["names"].values()) if isinstance(self.data["names"], dict) else self.data["names"]}
        self.im_dir = self.hub_dir / "images"
        self.im_dir.mkdir(parents=True, exist_ok=True)

    def _find_images(self, split):
        path = self.data.get(split)
        if not path:
            return []
        path = Path(path)
        if path.is_file():
            return [Path(x) for x in path.read_text().splitlines() if x.strip()]
        return list(path.rglob("*.*"))

    def get_json(self, save=False, verbose=False):
        stats = self.stats.copy()
        for split in ("train", "val", "test"):
            files = self._find_images(split)
            if not files:
                continue
            dataset = LoadImagesAndLabels(files, augment=False, rect=False, batch_size=1, prefix=f"{split}: ")
            instance_stats = []
            image_stats = []
            for im_file, labels in zip(dataset.im_files, dataset.labels):
                image_stats.append({"file": Path(im_file).name, "shape": dataset.shapes[dataset.im_files.index(im_file)].tolist(), "labels": labels.tolist()})
                instance_stats.extend(labels.tolist())
            stats[split] = {"images": len(dataset), "instances": len(instance_stats), "image_stats": image_stats, "instance_stats": instance_stats}
        if save:
            import json
            self.hub_dir.mkdir(parents=True, exist_ok=True)
            with open(self.hub_dir / "stats.json", "w") as f:
                json.dump(stats, f)
        if verbose:
            LOGGER.info(stats)
        return stats

    def process_images(self):
        for split in ("train", "val", "test"):
            for file in self._find_images(split):
                if file.suffix[1:].lower() in IMG_FORMATS:
                    destination = self.im_dir / split / file.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not destination.exists():
                        shutil.copy2(file, destination)
        return self.hub_dir