# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg587::matplotlib.pyplot.subplots+matplotlib.pyplot.subplots_adjust+numpy.arange
# name: matplotlib_numpy_primitive
# summary: Uses matplotlib.pyplot.subplots, matplotlib.pyplot.subplots_adjust, numpy.arange across 2 repos
# anchor_symbols: ['matplotlib.pyplot.subplots', 'matplotlib.pyplot.subplots_adjust', 'numpy.arange']
# observed in 2 repos: ['google__uncertainty-baselines', 'pykale__pykale']...

# --- from pykale__pykale::kale/interpret/uncertainty_utils.py::plot_uncertainty_correlation ---
def plot_uncertainty_correlation(
    uncertainties: np.ndarray,
    analysis_results: Dict[str, Any],
    quantile_thresholds: List[float],
    colormap: str = "Set1",
    to_log: bool = False,
    save_path: Optional[str] = None,
    show: bool = False,
    font_size: int = 25,
    **fig_kwargs: Any,
) -> None:
    """
    Create a comprehensive visualization of uncertainty-error correlation analysis.

    Args:
        uncertainties (np.ndarray): Array of uncertainty estimates
        analysis_results (Dict[str, Any]): Results from analyze_uncertainty_correlation
        quantile_thresholds (List[float]): Quantile threshold values for segmentation
        colormap (str, optional): Matplotlib colormap name for segment coloring.
            Defaults to "Set1".
        to_log (bool, optional): Whether to use logarithmic scale for both axes.
            Defaults to False.
        save_path (Optional[str], optional): File path to save the plot. Defaults to None.
        show (bool, optional): Whether to show the figure. Defaults to False.
        font_size (int, optional): Font size for all text elements. Defaults to 25.
        **fig_kwargs: Additional keyword arguments for figure creation and styling:
            - figsize (tuple): Figure size in inches (default: (16, 8))
            - save_dpi (int): Dots per inch for resolution (default: 600)
            - show_dpi (int): Dots per inch for the shown figure (default: 100)
            - bottom (float): Bottom margin for subplots_adjust (default: 0.2)
            - left (float): Left margin for subplots_adjust (default: 0.15)
            - Any other matplotlib figure parameters

    Returns:
        None: Creates and displays or saves the plot

    Raises:
        ValueError: If input parameters are invalid
        KeyError: If analysis_results is missing required keys
    """
    # Extract figure-specific kwargs with defaults
    figsize = fig_kwargs.pop("figsize", (16, 8))
    bottom = fig_kwargs.pop("bottom", 0.2)
    left = fig_kwargs.pop("left", 0.15)

    # Initialize plot
    fig, ax = plt.subplots(figsize=figsize, **fig_kwargs)

    # Handle color configuration
    colors = colormaps.get_cmap(colormap)(np.arange(3))
    scatter_color = colors[2]

    # Set axis scales
    if to_log:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
    ax.set_xlim(max(uncertainties), min(uncertainties))

    # Plot bootstrap confidence bands
    _plot_bootstrap_confidence_bands(ax, analysis_results["bootstrap_models"], uncertainties)

    # Plot scatter points
    _plot_scatter_points(ax, uncertainties, analysis_results["scaled_errors"], scatter_color)

    # Plot piecewise segments
    _plot_piecewise_segments(ax, analysis_results["piecewise_model"], uncertainties, quantile_thresholds, colors[:2])

    # Setup plot formatting
    _setup_plot_formatting(
        ax, analysis_results["bin_label_locs"], quantile_thresholds, analysis_results["correlations"], font_size
    )

    # Adjust layout
    plt.subplots_adjust(bottom=bottom, left=left)

    # Save or show plot using all remaining fig_kwargs
    save_or_show_plot(save_path=save_path, show=show, fig_size=figsize)

# --- from google__uncertainty-baselines::baselines/diabetic_retinopathy_detection/utils/plot_utils.py::plot_retention_curves ---
def plot_retention_curves(distribution_shift_name,
                          dataset_to_model_results,
                          plot_dir: str,
                          no_oracle=True,
                          cutoff_perc=0.99):
  """Plot retention curves for a given distributional shift task and

  corresponding results.

  Args:
    distribution_shift_name: str, distribution shift used to compute results.
    dataset_to_model_results: Dict, results for each evaluation dataset.
    plot_dir: str, where to store plots.
    no_oracle: bool, if True, converts retention array that is computed with an
      Oracle Collaborative metric to remove the contribution of the oracle.
    cutoff_perc: float, specifies at which proportion the curves should be
      cutoff.
  """
  set_matplotlib_constants()
  retention_types = [
      'retention_accuracy_arr', 'retention_nll_arr', 'retention_auroc_arr',
      'retention_auprc_arr'
  ]

  datasets = list(sorted(list(dataset_to_model_results.keys())))

  for dataset in datasets:
    dataset_results = dataset_to_model_results[dataset]
    for tuning_domain in ['indomain', 'joint']:
      for retention_type in retention_types:
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.20, bottom=0.20)

        retention_name = RETENTION_ARR_TO_FULL_NAME[retention_type]
        oracle_str = 'no_oracle' if no_oracle else 'oracle'
        plot_name = (f'retention-{distribution_shift_name}-{dataset}'
                     f'-{tuning_domain}-{retention_name}-{oracle_str}')

        model_names = []
        for ((mt, k, is_d, key_tuning_domain, n_mc),
             model_dict) in dataset_results.items():
          if tuning_domain != key_tuning_domain:
            continue
          model_name = get_model_name((mt, k, is_d, key_tuning_domain, n_mc))
          model_names.append(model_name)

          # Subsample the array to ~500 points
          retention_arr = np.array(model_dict[retention_type])

          if no_oracle:
            prop_expert = np.arange(
                retention_arr.shape[1]) / retention_arr.shape[1]
            prop_model = 1 - prop_expert
            retention_arr = (retention_arr - prop_expert) / prop_model

          if retention_arr.shape[1] > 500:
            subsample_factor = max(2, int(retention_arr.shape[1] / 500))
            retention_arr = retention_arr[:, ::subsample_factor]

          retain_percs = np.arange(
              retention_arr.shape[1]) / retention_arr.shape[1]
          n_seeds = retention_arr.shape[0]
          mean = np.mean(retention_arr, axis=0)
          std_err = np.std(retention_arr, axis=0) / np.sqrt(n_seeds)

          if cutoff_perc is not None and 'accuracy' in retention_type:
            retain_percs = retain_percs[:-100]
            mean = mean[:-100]
            std_err = std_err[:-100]

          if 'retention_nll_arr' == retention_type:
            cutoff_index = int(retain_percs.shape[0] * 0.95)
            retain_percs = retain_percs[:cutoff_index]
            mean = mean[:cutoff_index]
            std_err = mean[:cutoff_index]

          color, linestyle = get_colors_and_linestyle(
              MODEL_TYPE_TO_FULL_NAME[(mt, k > 1)])

          # Visualize mean with standard error
          ax.plot(
              retain_percs,
              mean,
              label=model_name,
              color=color,
              linestyle=linestyle)
          ax.fill_between(
              retain_percs,
              mean - std_err,
              mean + std_err,
              color=color,
              alpha=0.25)
          ax.set(
              xlabel='Proportion of Cases Referred to Expert',
              ylabel=retention_name)
          fig.tight_layout()

        if isinstance(plot_dir, str):
          os.makedirs(plot_dir, exist_ok=True)
          metric_plot_path = os.path.join(plot_dir, f'{plot_name}.pdf')
          fig.savefig(metric_plot_path, transparent=True, dpi=300, format='pdf')
          logging.info(f'Saved retention plot for distribution shift '
                       f'{distribution_shift_name},'
                       f'dataset {dataset}, '
                       f'tuning domain {tuning_domain}, '
                       f'metric {retention_type}, '
                       f'models {model_names}, '
                       f'oracle setting {oracle_str} to {metric_plot_path}.')

        print(plot_name)
        # plt.show()
