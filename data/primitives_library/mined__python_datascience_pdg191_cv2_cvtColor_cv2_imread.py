# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg191::cv2.cvtColor+cv2.imread
# name: cv2_primitive
# summary: Uses cv2.cvtColor, cv2.imread across 3 repos
# anchor_symbols: ['cv2.cvtColor', 'cv2.imread']
# observed in 3 repos: ['deepchecks__deepchecks', 'jasmcaus__caer', 'microsoft__RD-Agent']...

# --- from jasmcaus__caer::tests/color/test_gray.py::test_gray2yuv ---
def test_gray2yuv():
    cv_gray = cv.imread(tens_path)
    cv_gray = cv.cvtColor(cv_gray, cv.COLOR_BGR2GRAY)
    yuv = caer.gray2yuv(cv_gray)

    assert len(yuv.shape) == 3
    assert isinstance(yuv, caer.Tensor)
    assert yuv.is_yuv()

# --- from jasmcaus__caer::tests/color/test_gray.py::test_gray2luv ---
def test_gray2luv():
    cv_gray = cv.imread(tens_path)
    cv_gray = cv.cvtColor(cv_gray, cv.COLOR_BGR2GRAY)
    luv = caer.gray2luv(cv_gray)

    assert len(luv.shape) == 3
    assert isinstance(luv, caer.Tensor)
    assert luv.is_luv()

# --- from microsoft__RD-Agent::test/notebook/testfiles/main2.py::main.load_img_as_numpy_with_mask ---
def load_img_as_numpy_with_mask(filepath):
        try:
            img_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(f"cv2.imread failed for {filepath}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mask = green_mask(img_bgr)
            img4 = np.concatenate([img_rgb, mask*255], axis=2)
            return img4
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return np.zeros((32, 32, 4), dtype=np.uint8)

# --- from microsoft__RD-Agent::test/notebook/testfiles/main_missing_sections.py::CactusDataset.__getitem__ ---
def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.id2path[img_id]
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Cannot read image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        if self.labels is not None:
            label = self.labels[idx]
            return image, label, img_id
        else:
            return image, img_id

# --- from deepchecks__deepchecks::deepchecks/vision/vision_data/simple_classification_data.py::SimpleClassificationDataset.__getitem__ ---
def __getitem__(self, index: int) -> t.Tuple[np.ndarray, int]:
        """Get the image and label at the given index."""
        image_file = self.images[index]
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = self.classes_map[image_file.parent.name]

        if self.transforms is not None:
            transformed = self.transforms(image=image, target=target)
            image, target = transformed['image'], transformed['target']
        else:
            if self.transform is not None:
                image = self.transform(image=image)['image']
            if self.target_transform is not None:
                target = self.target_transform(target)

        return image, target

# --- from deepchecks__deepchecks::deepchecks/vision/datasets/detection/coco_utils.py::get_image_and_label ---
def get_image_and_label(image_file, label_file, transforms=None):
    """Get image and label in correct format for models from file paths."""
    opencv_image = cv2.imread(str(image_file))
    pil_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
    if label_file is not None and label_file.exists():
        img_labels = [l.split() for l in label_file.open('r').read().strip().splitlines()]
        img_labels = np.array(img_labels, dtype=np.float32)
    else:
        img_labels = np.zeros((0, 5), dtype=np.float32)

    # Transform x,y,w,h in yolo format (x, y are of the image center, and coordinates are normalized) to standard
    # x,y,w,h format, where x,y are of the top left corner of the bounding box and coordinates are absolute.
    bboxes = []
    for label in img_labels:
        x, y, w, h = label[1:]
        # Note: probably the normalization loses some accuracy in the coordinates as it truncates the number,
        # leading in some cases to `y - h / 2` or `x - w / 2` to be negative
        bboxes.append(np.array([
            max((x - w / 2) * pil_image.width, 0),
            max((y - h / 2) * pil_image.height, 0),
            w * pil_image.width,
            h * pil_image.height,
            label[0]
        ]))

    if transforms is not None:
        # Albumentations accepts images as numpy and bboxes in defined format + class at the end
        transformed = transforms(image=np.array(pil_image), bboxes=bboxes)
        pil_image = Image.fromarray(transformed['image'])
        bboxes = transformed['bboxes']

    return pil_image, bboxes
