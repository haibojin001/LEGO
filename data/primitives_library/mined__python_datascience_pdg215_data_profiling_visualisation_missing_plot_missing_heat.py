# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg215::data_profiling.visualisation.missing.plot_missing_heatmap+numpy.triu_indices_from+numpy.var
# name: data_profiling_numpy_primitive
# summary: Uses data_profiling.visualisation.missing.plot_missing_heatmap, numpy.triu_indices_from, numpy.var, numpy.zeros_like across 2 repos
# anchor_symbols: ['data_profiling.visualisation.missing.plot_missing_heatmap', 'numpy.triu_indices_from', 'numpy.var', 'numpy.zeros_like']
# observed in 2 repos: ['Data-Centric-AI-Community__fg-data-profiling', 'ploomber__sklearn-evaluation']...

# --- from Data-Centric-AI-Community__fg-data-profiling::src/data_profiling/model/pandas/missing_pandas.py::missing_heatmap ---
def missing_heatmap(config: Settings, df: pd.DataFrame) -> str:
    # Remove completely filled or completely empty variables.
    columns = [i for i, n in enumerate(np.var(df.isnull(), axis="rows")) if n > 0]
    df = df.iloc[:, columns]

    # Create and mask the correlation matrix. Construct the base heatmap.
    corr_mat = df.isnull().corr()
    mask = np.zeros_like(corr_mat)
    mask[np.triu_indices_from(mask)] = True
    return plot_missing_heatmap(
        config, corr_mat=corr_mat, mask=mask, columns=list(df.columns)
    )

# --- from Data-Centric-AI-Community__fg-data-profiling::src/data_profiling/model/spark/missing_spark.py::missing_heatmap ---
def missing_heatmap(config: Settings, df: DataFrame) -> str:
    df = MissingnoBarSparkPatch(df, columns=df.columns, original_df_size=df.count())

    # Remove completely filled or completely empty variables.
    columns = [i for i, n in enumerate(np.var(df.isnull(), axis="rows")) if n > 0]
    df = df.iloc[:, columns]

    # Create and mask the correlation matrix. Construct the base heatmap.
    corr_mat = df.isnull().corr()
    mask = np.zeros_like(corr_mat)
    mask[np.triu_indices_from(mask)] = True
    return plot_missing_heatmap(
        config, corr_mat=corr_mat, mask=mask, columns=list(df.columns)
    )

# --- from ploomber__sklearn-evaluation::src/sklearn_evaluation/plot/feature_ranking.py::Rank2D._draw ---
def _draw(self):
        """
        Draws the heatmap of the ranking matrix of variables.
        """

        title = "{} Ranking of {} Features".format(
            self.algorithm.title(), len(self.features_)
        )
        self.ax.set_title(title)

        # Set the axes aspect to be equal
        self.ax.set_aspect("equal")

        # Generate a mask for the upper triangle
        mask = np.zeros_like(self.ranks_, dtype=bool)
        mask[np.triu_indices_from(mask)] = True

        # Draw the heatmap
        data = np.ma.masked_where(mask, self.ranks_)
        mesh = self.ax.pcolormesh(data, cmap=self.colormap, vmin=-1, vmax=1)

        # Set the Axis limits
        self.ax.set(xlim=(0, data.shape[1]), ylim=(0, data.shape[0]))

        # Add the colorbar
        cb = self.ax.figure.colorbar(mesh, None, self.ax, fraction=0.046, pad=0.04)
        cb.outline.set_linewidth(0)

        # Reverse the rows to get the lower left triangle
        self.ax.invert_yaxis()

        # Add ticks and tick labels
        self.ax.set_xticks(np.arange(len(self.ranks_)) + 0.5)
        self.ax.set_yticks(np.arange(len(self.ranks_)) + 0.5)
        self.ax.set_xticklabels(self.features_, rotation=90)
        self.ax.set_yticklabels(self.features_)
