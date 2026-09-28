# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg141::cv2.cvtColor+cv2.imwrite
# name: cv2_primitive
# summary: Uses cv2.cvtColor, cv2.imwrite across 2 repos
# anchor_symbols: ['cv2.cvtColor', 'cv2.imwrite']
# observed in 2 repos: ['OML-Team__open-metric-learning', 'microsoft__nni']...

# --- from microsoft__nni::examples/trials/kaggle-tgs-salt/postprocessing.py::save_pseudo_label_masks ---
def save_pseudo_label_masks(submission_file):
    df = pd.read_csv(submission_file, na_filter=False)
    print(df.head())

    img_dir = os.path.join(settings.TEST_DIR, 'masks')

    for i, row in enumerate(df.values):
        decoded_mask = run_length_decoding(row[1], (101,101))
        filename = os.path.join(img_dir, '{}.png'.format(row[0]))
        rgb_mask = cv2.cvtColor(decoded_mask,cv2.COLOR_GRAY2RGB)
        print(filename)
        cv2.imwrite(filename, decoded_mask)
        if i % 100 == 0:
            print(i)

# --- from OML-Team__open-metric-learning::tests/test_oml/test_utils/test_readers.py::test_readers ---
def test_readers(img_format: str, num_channels: int, no_compression: bool) -> None:
    shape_hw = (333, 257)
    dummy_image = np.random.randint(0, 255, (*shape_hw, num_channels), dtype=np.uint8)

    fname_image = str(TMP_PATH / f"img_test_readers.{img_format}")

    if img_format == "png" and no_compression:
        cv2.imwrite(fname_image, cv2.cvtColor(dummy_image, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        cv2.imwrite(fname_image, cv2.cvtColor(dummy_image, cv2.COLOR_RGBA2BGRA))

    image_cv2_from_path = imread_cv2(fname_image)
    image_pil_from_path = np.array(imread_pillow(fname_image))

    with open(fname_image, "rb") as fin:
        image_bytes = fin.read()

    image_cv2_from_bytes = imread_cv2(image_bytes)
    image_pil_from_bytes = np.array(imread_pillow(image_bytes))

    for image in [image_cv2_from_path, image_pil_from_path, image_cv2_from_bytes, image_pil_from_bytes]:
        assert image.shape == (*shape_hw, 3)
        if no_compression:
            assert np.array_equal(image, dummy_image[:, :, :3]), (image == dummy_image[:, :, :3]).mean()
        else:
            assert np.array_equal(image, image_cv2_from_path), (image == image_cv2_from_path).mean()
