# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg77::numpy.ones_like+numpy.repeat+pandas.DataFrame
# name: numpy_pandas_primitive
# summary: Uses numpy.ones_like, numpy.repeat, pandas.DataFrame across 2 repos
# anchor_symbols: ['numpy.ones_like', 'numpy.repeat', 'pandas.DataFrame']
# observed in 2 repos: ['BiomedSciAI__causallib', 'recommenders-team__recommenders']...

# --- from BiomedSciAI__causallib::causallib/simulation/CausalSimulator3.py::CausalSimulator3._treatment_odds_ratio ---
def _treatment_odds_ratio(x_continuous, prob_category, snr):
        """
        Assign treatment proportional to the odds ratio of the categories.
        Each category is assigned with it's odds ratio independently (based on logistic function) and are later sampled
        proportional to these odds ratio.
        Args:
            x_continuous (pd.Series): Aggregated signal (a scalar per sample) based on the variable's predecessor
                                      variables.
            prob_category (pd.Series): Probability vector the size of number of treatment categories with every entry is
                                       the corresponding probability of that category.
            snr (float) - signal to noise ratio.

        Returns:
            (pd.DataFrame, pd.DataFrame): 2-element tuple containing:

            - **treatment** (*pd.Series*): Treatment assignment for each sample.
            - **propensity** (*pd.DataFrame*): The marginal conditional probability of treatment given covariates.
                                               A DataFrame shaped (num_samples x num_of_possible_treatment_categories).
        """
        x_continuous = x_continuous.astype(float)
        index_names = x_continuous.index
        columns_names = prob_category.index
        propensity = pd.DataFrame(index=index_names, columns=columns_names)
        # start with filling up the odds ratio:
        for cur_category, p in prob_category.items():
            t = x_continuous.quantile(p, interpolation="higher")
            cur_propensity = (1.0 / (1 + np.exp((x_continuous - np.repeat(t, x_continuous.size)))))  # type: pd.Series
            cur_propensity = cur_propensity.div(np.ones_like(cur_propensity) - cur_propensity)
            cur_propensity += np.abs(np.random.normal(loc=0.0, scale=1 - snr, size=cur_propensity.size))
            # cur_propensity += np.random.exponential(scale=np.sqrt(snr), size=cur_propensity.size)
            propensity.loc[:, cur_category] = cur_propensity

        # normalize into probabilities:
        propensity = propensity.div(propensity.sum(axis="columns"), axis="rows")
        # treatment assignment is drawn according to marginal propensities:
        treatment = CausalSimulator3._sample_from_row_stochastic_matrix(propensity)
        return propensity, treatment

# --- from recommenders-team__recommenders::recommenders/models/sar/sar_singlenode.py::SARSingleNode.get_item_based_topk ---
def get_item_based_topk(self, items, top_k=10, sort_top_k=True):
        """Get top K similar items to provided seed items based on similarity metric defined.
        This method will take a set of items and use them to recommend the most similar items to that set
        based on the similarity matrix fit during training.
        This allows recommendations for cold-users (unseen during training), note - the model is not updated.

        The following options are possible based on information provided in the items input:
        1. Single user or seed of items: only item column (ratings are assumed to be 1)
        2. Single user or seed of items w/ ratings: item column and rating column
        3. Separate users or seeds of items: item and user column (user ids are only used to separate item sets)
        4. Separate users or seeds of items with ratings: item, user and rating columns provided

        Args:
            items (pandas.DataFrame): DataFrame with item, user (optional), and rating (optional) columns
            top_k (int): number of top items to recommend
            sort_top_k (bool): flag to sort top k results

        Returns:
            pandas.DataFrame: sorted top k recommendation items
        """

        # convert item ids to indices
        item_ids = np.asarray(
            list(
                map(
                    lambda item: self.item2index.get(item, np.NaN),
                    items[self.col_item].values,
                )
            )
        )

        # if no ratings were provided assume they are all 1
        if self.col_rating in items.columns:
            ratings = items[self.col_rating]
        else:
            ratings = pd.Series(np.ones_like(item_ids))

        # create local map of user ids
        if self.col_user in items.columns:
            test_users = items[self.col_user]
            user2index = {x[1]: x[0] for x in enumerate(items[self.col_user].unique())}
            user_ids = test_users.map(user2index)
        else:
            # if no user column exists assume all entries are for a single user
            test_users = pd.Series(np.zeros_like(item_ids))
            user_ids = test_users
        n_users = user_ids.drop_duplicates().shape[0]

        # generate pseudo user affinity using seed items
        pseudo_affinity = sparse.coo_matrix(
            (ratings, (user_ids, item_ids)), shape=(n_users, self.n_items)
        ).tocsr()

        # calculate raw scores with a matrix multiplication
        test_scores = pseudo_affinity.dot(self.item_similarity)

        # remove items in the seed set so recommended items are novel
        test_scores[user_ids, item_ids] = -np.inf

        top_items, top_scores = get_top_k_scored_items(
            scores=test_scores, top_k=top_k, sort_top_k=sort_top_k
        )

        df = pd.DataFrame(
            {
                self.col_user: np.repeat(
                    test_users.drop_duplicates().values, top_items.shape[1]
                ),
                self.col_item: [self.index2item[item] for item in top_items.flatten()],
                self.col_prediction: top_scores.flatten(),
            }
        )

        # drop invalid items
        return df.replace(-np.inf, np.nan).dropna()
