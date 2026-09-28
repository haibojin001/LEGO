# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg670::matplotlib.pyplot.axis+matplotlib.pyplot.figure+matplotlib.pyplot.imshow
# name: matplotlib_primitive
# summary: Uses matplotlib.pyplot.axis, matplotlib.pyplot.figure, matplotlib.pyplot.imshow, matplotlib.pyplot.show across 3 repos
# anchor_symbols: ['matplotlib.pyplot.axis', 'matplotlib.pyplot.figure', 'matplotlib.pyplot.imshow', 'matplotlib.pyplot.show', 'matplotlib.pyplot.subplot', 'matplotlib.pyplot.title']
# observed in 3 repos: ['JosephLai241__URS', 'OML-Team__open-metric-learning', 'ahmetozlu__tensorflow_object_counting_api']...

# --- from JosephLai241__URS::urs/analytics/Wordcloud.py::SetUpWordcloud.modify_wordcloud ---
def modify_wordcloud(wc: WordCloud):
        """
        Further modify wordcloud preferences.

        :param WordCloud wc: The `WordCloud` instance.

        :returns: A `matplotlib.pyplot` instance.
        :rtype: `matplotlib.pyplot`
        """

        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")

        return plt

# --- from ahmetozlu__tensorflow_object_counting_api::mask_rcnn_counting_api/spaghetti_counter_training/training/mrcnn/visualize.py::display_images ---
def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()

# --- from ahmetozlu__tensorflow_object_counting_api::mask_rcnn_counting_api/spaghetti_counter_training/mrcnn/visualize.py::display_images ---
def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()

# --- from OML-Team__open-metric-learning::oml/retrieval/retrieval_results.py::RetrievalResults.visualize_with_functions ---
def visualize_with_functions(
        self,
        query_ids: List[int],
        visualize_query_fn: Callable[[int, TColor], np.ndarray],
        visualize_gallery_fn: Callable[[int, TColor], np.ndarray],
        n_galleries_to_show: int = 5,
        n_gt_to_show: int = N_GT_SHOW_EMBEDDING_METRICS,
        show: bool = False,
    ) -> plt.Figure:
        """
        Args:
            query_ids: Query indices within the range of ``(0, n_query - 1)``.
            visualize_query_fn: Function plotting ``i-th`` query with respect to the given color.
            visualize_gallery_fn: Function plotting ``j-th`` gallery with respect to the given color.
            n_galleries_to_show: Number of closest gallery items to show.
            n_gt_to_show: Number of ground truth gallery items to show for reference (if available).
            show: Set ``True`` to instantly visualize the resulted figure.

        """

        max_presented_galleries = max(len(self.retrieved_ids[iq]) for iq in query_ids)
        n_galleries_to_show = min(n_galleries_to_show, max_presented_galleries)
        n_gt_to_show = n_gt_to_show if (self.gt_ids is not None) else 0

        fig = plt.figure(figsize=(16, 16 / (n_galleries_to_show + n_gt_to_show + 1) * len(query_ids)))
        n_rows, n_cols = len(query_ids), n_galleries_to_show + 1 + n_gt_to_show

        # iterate over queries
        for i, query_idx in enumerate(query_ids):

            plt.subplot(n_rows, n_cols, i * (n_galleries_to_show + 1 + n_gt_to_show) + 1)

            img = visualize_query_fn(query_idx, BLUE)

            plt.imshow(img)
            plt.title(f"Query #{query_idx}")
            plt.axis("off")

            # iterate over retrieved items
            for j, ret_idx in enumerate(self.retrieved_ids[query_idx][:n_galleries_to_show]):
                if self.gt_ids is not None:
                    color = GREEN if ret_idx in self.gt_ids[query_idx] else RED
                else:
                    color = BLACK

                plt.subplot(n_rows, n_cols, i * (n_galleries_to_show + 1 + n_gt_to_show) + j + 2)
                img = visualize_gallery_fn(ret_idx, color)

                plt.title(f"Gallery #{ret_idx} - {round(self.distances[query_idx][j].item(), 3)}")
                plt.imshow(img)
                plt.axis("off")

            if self.gt_ids is not None:

                for k, gt_idx in enumerate(self.gt_ids[query_idx][:n_gt_to_show]):
                    plt.subplot(
                        n_rows, n_cols, i * (n_galleries_to_show + 1 + n_gt_to_show) + k + n_galleries_to_show + 2
                    )

                    img = visualize_gallery_fn(gt_idx, GRAY)

                    plt.title("GT")
                    plt.imshow(img)
                    plt.axis("off")

        fig.tight_layout()

        if show:
            fig.show()

        return fig
