def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        """
        Fit the SAMME Adaboost model on the entire dataset each iteration.

        Parameters
        ----------
        X_train : ndarray (n_samples, n_features)
        y_train : ndarray (n_samples,)
        X_valid, y_valid : optional validation set
        """
        n_samples = X_train.shape[0]

        if self.n_classes is None:
            self.n_classes = len(np.unique(y_train))

        # Initialize sample weights
        w = np.ones(n_samples) / n_samples

        # Reset stored attributes
        self.alphas.clear()
        self.models.clear()
        self.train_accuracies.clear()
        self.train_losses.clear()
        self.validation_accuracies.clear()
        self.validation_losses.clear()
        self.iterations.clear()

        for m in range(1, self.n_rounds + 1):
            # 1. Create and train a new weak learner on full data with sample_weight=w
            model = self.base_learner_factory()
            if m==1:
                    # Generate random indices for 10% of the training data
                    sample_size = int(0.1 * X_train.shape[0])
                    sampled_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)

                    # Select the subset of data
                    X_train_sample = X_train[sampled_indices]
                    y_train_sample = y_train[sampled_indices]

                    # Fit the model on the 10% subset for a weak learner
                    model.fit(X_train_sample, y_train_sample)
            else:
                model.fit(X_train, y_train,sample_weight=w)
                
               
           

            # 2. Compute weighted error
            y_pred_train = model.predict(X_train)
            misclassified = (y_pred_train != y_train).astype(float)
            err_m = np.sum(w * misclassified) / np.sum(w) + 1e-15

            # Early stop if 1 - error <= 1/K
            if 1 - err_m <= 1/self.n_classes + 1e-15:
                print(f"Stopping early at round {m}, weighted error = {err_m:.4f}")
                break

            # 3. Compute alpha_m
            #    alpha_m = ln((1 - err_m)/err_m) + ln(K-1)
            alpha_m = np.log((1 - err_m) / err_m) + np.log(self.n_classes - 1)

            # 4. Update sample weights
            #    w[i] *= exp(alpha_m) if y_pred != y_true
            #    then re-normalize
            w *= np.exp(alpha_m * misclassified)
            w /= np.sum(w)

            # Store the model and alpha
            self.alphas.append(alpha_m)
            self.models.append(model)
            self.iterations.append(m)

            # 5. Track metrics on training
            y_pred_train = self.predict(X_train)
            train_accuracy = np.mean(y_pred_train == y_train)
            self.train_accuracies.append(train_accuracy)
            self.train_losses.append(err_m)  # Weighted training error

            if X_valid is not None and y_valid is not None:
                y_pred_val = self.predict(X_valid)
                val_accuracy = np.mean(y_pred_val == y_valid)
                self.validation_accuracies.append(val_accuracy)
                val_misclassified = (y_pred_val != y_valid).astype(float)
                val_loss = np.mean(val_misclassified)
                self.validation_losses.append(val_loss)

                print(
                    f"Round {m}: Weighted Train Err={err_m:.4f}, "
                    f"Train Acc={train_accuracy:.4f}, "
                    f"Val Err={val_loss:.4f}, Val Acc={val_accuracy:.4f}"
                )
            else:
                print(
                    f"Round {m}: Weighted Train Err={err_m:.4f}, "
                    f"Train Acc={train_accuracy:.4f}"
                )