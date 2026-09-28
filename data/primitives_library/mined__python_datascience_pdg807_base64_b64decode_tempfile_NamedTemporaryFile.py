# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg807::base64.b64decode+tempfile.NamedTemporaryFile
# name: base64_tempfile_primitive
# summary: Uses base64.b64decode, tempfile.NamedTemporaryFile across 2 repos
# anchor_symbols: ['base64.b64decode', 'tempfile.NamedTemporaryFile']
# observed in 2 repos: ['AgnostiqHQ__covalent', 'underneathall__pinferencia']...

# --- from AgnostiqHQ__covalent::covalent/_dispatcher_plugins/local.py::decode_b64_tar ---
def decode_b64_tar(b64_buffer: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".tar") as tar_file:
        tar_path = tar_file.name

    with open(tar_path, "wb") as tar_file:
        tar_file.write(base64.b64decode(b64_buffer.encode("utf-8")))

    return tar_path

# --- from underneathall__pinferencia::tests/e2e_tests/templates/test_image_to_text.py::test_success ---
def test_success(task, image_base64_string, page):
    # choose the return text model
    model = page.locator("text=invalid-task-model")
    model.click()
    return_text_model = page.locator("text=return-text-model")
    return_text_model.click()

    # locate the sidebar
    sidebar = page.locator('section[data-testid="stSidebar"]')

    # open the selector
    # here, instead of using:
    # task_selector = sidebar.locator(
    #     'div[data-baseweb="select"]:below(:text("Select the Task"))'
    # )
    # we choose to select the 'Text To Text' spefically, just in case
    # the task selection is clicked too fast and streamlit re-select the
    # default task of the model again.
    task_selector = sidebar.locator("text='Text To Text'")
    task_selector.click()

    # choose the task
    task = page.locator("li[role='option']").locator(f"text='{task}'")
    task_selector.wait_for(timeout=10000)
    task.click()

    main_div = page.locator("section.main")

    # upload
    with page.expect_file_chooser() as fc_info:
        page.click("text='Browse files'")

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".jpg") as f:
        # create a temporary image file and write the image bytes
        f.write(base64.b64decode(image_base64_string))

        # flush the content to disk
        f.flush()

        # choose the created file
        file_chooser = fc_info.value
        file_chooser.set_files(f.name)

        page.click("text='Upload and Run'")

        # wait for the result
        result = main_div.locator('div.stAlert:has-text("abcdefg")')
        result.wait_for(timeout=10000)

        assert result.count() == 1

# --- from underneathall__pinferencia::tests/e2e_tests/templates/test_image_to_image.py::test_success ---
def test_success(task, image_base64_string, page):
    # choose the return text model
    model = page.locator("text=invalid-task-model")
    model.click()
    return_image_model = page.locator("text=return-image-model")
    return_image_model.click()

    # locate the sidebar
    sidebar = page.locator('section[data-testid="stSidebar"]')

    # open the selector
    # here, instead of using:
    # task_selector = sidebar.locator(
    #     'div[data-baseweb="select"]:below(:text("Select the Task"))'
    # )
    # we choose to select the 'Text To Text' spefically, just in case
    # the task selection is clicked too fast and streamlit re-select the
    # default task of the model again.
    task_selector = sidebar.locator("text='Text To Image'")
    task_selector.wait_for(timeout=10000)
    task_selector.click()

    # choose the task
    task = page.locator("li[role='option']").locator(f"text='{task}'")
    task.click()

    main_div = page.locator("section.main")

    # upload
    with page.expect_file_chooser() as fc_info:
        page.click("text='Browse files'")

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".jpg") as f:
        # create a temporary image file and write the image bytes
        f.write(base64.b64decode(image_base64_string))

        # flush the content to disk
        f.flush()

        # choose the created file
        file_chooser = fc_info.value
        file_chooser.set_files(f.name)

        page.click("text='Upload and Run'")

        # wait for the result
        result_column = main_div.locator('div[data-testid="column"]:has-text("Result")')
        result = result_column.locator('div[data-testid="stImage"]')
        result.wait_for(timeout=10000)

        assert result.count() == 1
