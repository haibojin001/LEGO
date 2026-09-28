def calculate_gini(df):
    """Calculates Gini Impurity for each feature and its values in the DataFrame."""
    label_col = df.columns[-1]  
    gini_result = {}

    for feature in df.columns[:-1]: # [:-1] Exclude the last column
        print("\033[1;32m"f"\nfor '{feature}'\033[0m")

        impurity_values = []
        for value in df[feature].unique(): # "unique()" removes duplicated values(from pandas lib)
            subset_labels = df[df[feature] == value][label_col] # Labels for this value
            total = len(subset_labels)

            class_counts = subset_labels.value_counts() # Counts 'yes' and 'no'
            prob_yes = 0
            if "yes" in class_counts:
                prob_yes = class_counts["yes"] / total
            prob_no = 1 - prob_yes
            # The Gini mathematical formula
            impurity = 1 - (prob_yes ** 2 + prob_no ** 2) 
            impurity_values.append(impurity)

            print("\033[1;33m"f"    Value '{value}':""\033[0m" f" [Gini = {impurity:.4f}], [Prob = {prob_yes:.3f}]")

        gini_result[feature] = sum(impurity_values) / len(impurity_values)

    print("\033[0;32m""-" * 55 + "\033[0m") # Graphical separator line in the terminal
    return gini_result