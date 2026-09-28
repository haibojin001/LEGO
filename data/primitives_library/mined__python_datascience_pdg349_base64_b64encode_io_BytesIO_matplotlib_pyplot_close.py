# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg349::base64.b64encode+io.BytesIO+matplotlib.pyplot.close
# name: base64_io_primitive
# summary: Uses base64.b64encode, io.BytesIO, matplotlib.pyplot.close, matplotlib.pyplot.figure across 2 repos
# anchor_symbols: ['base64.b64encode', 'io.BytesIO', 'matplotlib.pyplot.close', 'matplotlib.pyplot.figure', 'matplotlib.pyplot.savefig', 'matplotlib.pyplot.tight_layout']
# observed in 2 repos: ['OML-Team__open-metric-learning', 'business-science__ai-data-science-team']...

# --- from business-science__ai-data-science-team::ai_data_science_team/tools/eda.py::visualize_missing.create_and_encode_plot ---
def create_and_encode_plot(plot_func, plot_name: str):
        plt.figure(figsize=(8, 6))
        # Call the missingno plotting function.
        plot_func(df)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

# --- from OML-Team__open-metric-learning::oml/utils/audios.py::_visualize_audio ---
def _visualize_audio(
    spec_repr: FloatTensor, color: TColor = BLACK, draw_bbox: bool = True, return_b64: bool = False
) -> Union[np.ndarray, str]:
    """
    Internal function to visualize an audio spectrogram.

    Args:
        spec_repr: The spectrogram representation.
        color: The color of the bounding box.
        draw_bbox: Whether to draw a bounding box around the spectrogram.
        return_b64: Whether to return the image as a base64 string.

    Returns:
        If return_b64 is ``False``, returns the image as an array.
        If return_b64 is ``True``, returns the image as a base64 string.
    """
    fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)

    # actual image
    ax.imshow(spec_repr, aspect="auto", origin="lower")

    # bbox and axes
    if draw_bbox:
        frame_thickness = 5
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(frame_thickness)
            ax.spines[axis].set_edgecolor([c / 255 for c in color])
    ax.set_xticks([])
    ax.set_yticks([])

    # drawing
    fig.canvas.draw()
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img: Union[np.ndarray, str] = (
        base64.b64encode(buf.getvalue()).decode("ascii") if return_b64 else np.array(Image.open(buf), dtype=np.uint8)
    )
    plt.close(fig)
    return img

# --- from OML-Team__open-metric-learning::oml/utils/audios.py::visualize_audio_with_player ---
def visualize_audio_with_player(
    audio: FloatTensor,
    spec_repr: FloatTensor,
    sample_rate: int,
    title: str = "",
    color: TColor = BLACK,
    draw_bbox: bool = True,
) -> str:
    """
    Visualize an audio spectral representation and provide an HTML string with an audio player.

    Args:
        audio: The audio waveform.
        spec_repr: The spectral representation.
        sample_rate: The sampling rate of the audio.
        title: Title of the output HTML block if needed.
        color: The color of the bounding box.
        draw_bbox: Whether to draw a bounding box around the spectral representation.

    Returns:
        An HTML string that contains the spectral representation image and an audio player.
    """
    import torchaudio

    image_base64 = _visualize_audio(spec_repr, color, draw_bbox, return_b64=True)  # type: ignore

    buf = BytesIO()
    torchaudio.save(buf, audio, sample_rate=sample_rate, format="wav")
    buf.seek(0)
    audio_base64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # generate HTML
    html = f"""
    <div style="margin: 10px; display: inline-block;">
        <div style="text-align: center; font-weight: bold; margin-bottom: 5px;">{title}</div>
        <div style="border:5px solid {color}; padding: 3px;">
            <img src="data:image/png;base64,{image_base64}" style="width:128px; height:128px;" alt="Mel Spectrogram" />
            <audio controls style="display: block; width: 100%; margin-top: 3px;">
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
        </div>
    </div>
    """

    return html

# --- from business-science__ai-data-science-team::ai_data_science_team/tools/eda.py::visualize_missing ---
def visualize_missing(
    data_raw: Annotated[dict, InjectedState("data_raw")], n_sample: int = None
) -> Tuple[str, Dict]:
    """
    Tool: visualize_missing
    Description:
        Missing value analysis using the missingno library. Generates a matrix plot, bar plot, and heatmap plot.

    Parameters:
    -----------
    data_raw : dict
        The raw data in dictionary format.
    n_sample : int, optional (default: None)
        The number of rows to sample from the dataset if it is large.

    Returns:
    -------
    Tuple[str, Dict]:
        content: A message describing the generated plots.
        artifact: A dict with keys 'matrix_plot', 'bar_plot', and 'heatmap_plot' each containing the
                  corresponding base64 encoded PNG image.
    """
    print("    * Tool: visualize_missing")

    try:
        import missingno as msno  # Ensure missingno is installed
    except ImportError:
        raise ImportError(
            "Please install the 'missingno' package to use this tool. pip install missingno"
        )

    import pandas as pd
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt

    # Create the DataFrame and sample if n_sample is provided.
    df = pd.DataFrame(data_raw)
    if n_sample is not None:
        df = df.sample(n=n_sample, random_state=42)

    # Dictionary to store the base64 encoded images for each plot.
    encoded_plots = {}

    # Define a helper function to create a plot, save it, and encode it.
    def create_and_encode_plot(plot_func, plot_name: str):
        plt.figure(figsize=(8, 6))
        # Call the missingno plotting function.
        plot_func(df)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # Create and encode the matrix plot.
    encoded_plots["matrix_plot"] = create_and_encode_plot(msno.matrix, "matrix")

    # Create and encode the bar plot.
    encoded_plots["bar_plot"] = create_and_encode_plot(msno.bar, "bar")

    # Create and encode the heatmap plot.
    encoded_plots["heatmap_plot"] = create_and_encode_plot(msno.heatmap, "heatmap")

    content = (
        "Missing data visualizations (matrix, bar, and heatmap) have been generated."
    )
    artifact = encoded_plots
    return content, artifact
