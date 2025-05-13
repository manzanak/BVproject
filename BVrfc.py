import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.metrics import classification_report, roc_curve, auc, ConfusionMatrixDisplay
import networkx as nx
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def label_comprehensive_recurrence_patterns(diagnosis_data):
    diagnosis_sorted = diagnosis_data.sort_values(['Subject_ID', 'Follow-up'])
    diagnosis_sorted['Recurrence_Type'] = 'No_BV'  # Default
    recurrence_subjects = {}  # Dictionary to track recurrence types by subject

    for subject_id, group in diagnosis_sorted.groupby('Subject_ID'):
        if len(group) < 2:  # Need at least 2 follow-ups
            continue

        bv_statuses = group['BV_status'].tolist()
        follow_ups = group['Follow-up'].tolist()

        # Initialize subject's recurrence status
        recurrence_subjects[subject_id] = {'has_recurrence': False, 'type': None}

        # Check if patient ever had BV
        if 1 not in bv_statuses:
            continue

        # Pattern 1: Any recurrence (BV positive → negative → positive again, at any point)
        neg_after_pos = False
        for i in range(1, len(bv_statuses)):
            # First check if there was a negative after positive
            if bv_statuses[i-1] == 1 and bv_statuses[i] == 0:
                neg_after_pos = True
            # Then check if there was a positive after that negative
            if neg_after_pos and bv_statuses[i] == 1:
                recurrence_subjects[subject_id] = {'has_recurrence': True, 'type': 'Any_Recurrence'}
                idx = group.index
                diagnosis_sorted.loc[idx, 'Recurrence_Type'] = 'Any_Recurrence'
                break

        # Pattern 2: Immediate recurrence (BV positive → negative → positive in consecutive follow-ups)
        for i in range(len(bv_statuses) - 2):
            if bv_statuses[i:i+3] == [1, 0, 1]:
                recurrence_subjects[subject_id] = {'has_recurrence': True, 'type': 'Immediate_Recurrence'}
                idx = group.index[i:i+3]
                diagnosis_sorted.loc[idx, 'Recurrence_Type'] = 'Immediate_Recurrence'
                break

        # Pattern 3: Delayed recurrence (BV positive → negative for at least 2 follow-ups → positive again)
        for i in range(len(bv_statuses) - 3):
            if bv_statuses[i] == 1 and bv_statuses[i+1] == 0 and bv_statuses[i+2] == 0 and 1 in bv_statuses[i+3:]:
                recurrence_subjects[subject_id] = {'has_recurrence': True, 'type': 'Delayed_Recurrence'}
                idx = group.index
                diagnosis_sorted.loc[idx, 'Recurrence_Type'] = 'Delayed_Recurrence'
                break

        # Pattern 4: No initial response (BV positive → still positive after treatment)
        for i in range(len(bv_statuses) - 1):
            if bv_statuses[i] == 1 and bv_statuses[i+1] == 1:
                recurrence_subjects[subject_id] = {'has_recurrence': True, 'type': 'No_Initial_Response'}
                idx = group.index[i:i+2]
                diagnosis_sorted.loc[idx, 'Recurrence_Type'] = 'No_Initial_Response'
                break

        # Pattern 5: Successful treatment (BV positive → negative for all subsequent follow-ups)
        if 1 in bv_statuses and all(status == 0 for status in bv_statuses[bv_statuses.index(1) + 1:]):
            recurrence_subjects[subject_id] = {'has_recurrence': False, 'type': 'Successful_Treatment'}
            idx = group.index
            diagnosis_sorted.loc[idx, 'Recurrence_Type'] = 'Successful_Treatment'

    return diagnosis_sorted, recurrence_subjects

def visualize_recurrence_patterns(recurrence_subjects_dict):
    """
    Creates a bar chart showing the distribution of different BV recurrence patterns.

    Parameters:
    -----------
    recurrence_subjects_dict : dict
        Dictionary mapping subject IDs to their recurrence patterns
    """
    # Count unique subjects with each pattern
    subject_pattern_counts = {}
    for subject, info in recurrence_subjects_dict.items():
        pattern = info['type']
        if pattern is None:
            continue
        if pattern not in subject_pattern_counts:
            subject_pattern_counts[pattern] = 0
        subject_pattern_counts[pattern] += 1

    print("Unique Subjects with Each Recurrence Pattern:")
    for pattern, count in subject_pattern_counts.items():
        print(f"{pattern}: {count}")

    # Visualize the distribution of recurrence patterns
    plt.figure(figsize=(12, 6))
    plt.bar(subject_pattern_counts.keys(), subject_pattern_counts.values())
    plt.title('Distribution of BV Recurrence Patterns Across Subjects', fontsize=14)
    plt.ylabel('Number of Subjects', fontsize=12)
    plt.xlabel('Recurrence Pattern', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def normalize_otus(patient_data, otu_names):
    """
    Creates a correlation matrix and co-occurrence network for a given patient's data.

    Args:
        patient_data: DataFrame containing OTU data for a single patient.
        otu_names: DataFrame mapping OTU IDs to taxonomic classifications.
        threshold: The correlation coefficient threshold for creating edges in the network.
        p_value_threshold: The p-value threshold for significance.

    Returns:
        A tuple containing the correlation matrix and the networkx graph.
    """

 # Select OTU columns
    total_reads = patient_data['total_reads'].astype(float)
    OTUs = patient_data.drop(columns=['Follow-up', 'Subject_ID', 'Sample_ID',
                                      'Amsel_discharge','Amsel_odor',
                                      'Amsel_clue','pH','Amsel_score',
                                      'Nugent_Score', 'total_reads',
                                      'BV_status'])
    OTUs = OTUs.apply(pd.to_numeric, errors='coerce') #This line converts all columns in OTUs to numeric, coercing any errors to NaN.

    # Normalize OTU reads
    OTUs = OTUs.div(total_reads, axis=0)

    # Using mean abundance
    abundance_threshold = 0.001
    otu_mean_abundance = OTUs.mean(axis=0)
    # display(otu_mean_abundance)

    # Filter OTUs
    filtered_otus = otu_mean_abundance[otu_mean_abundance >= abundance_threshold].index

    # Keep only the filtered OTUs in your normalized data
    filtered_data = OTUs[filtered_otus]

    # Convert the filtered data to numeric type before normalization
    filtered_data = filtered_data.apply(pd.to_numeric, errors='coerce') #This line converts all columns in filtered_data to numeric, coercing any errors to NaN.

    # Normalize OTU reads
    filtered_data = filtered_data.div(filtered_data.sum(axis=1), axis=0)
    
    return filtered_data

def plot_pca_microbiome(diagnosis_data, recurrence_dict):
    # Filter to baseline samples
    baseline = diagnosis_data[diagnosis_data['Follow-up'] == '1'].copy()
    baseline['Subject_ID'] = baseline['Subject_ID'].astype(str)

    # Assign recurrence label
    baseline['Has_Recurrence'] = baseline['Subject_ID'].map(
        {subj: 1 if info['has_recurrence'] else 0 for subj, info in recurrence_dict.items()}
    )
    baseline = baseline.dropna(subset=['Has_Recurrence'])

    # Get OTU features
    otu_columns = [col for col in baseline.columns if col.startswith('otu')]
    X = baseline[otu_columns].astype(float)

    # Normalize features (PCA assumes centered/standardized data)
    X_scaled = StandardScaler().fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    # Plot
    baseline['PC1'] = pcs[:, 0]
    baseline['PC2'] = pcs[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=baseline, x='PC1', y='PC2', hue='Has_Recurrence', palette='Set1', s=100)
    plt.title('PCA of Baseline Microbiome Composition')
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.legend(['Any Recurrence', 'No Recurrence'])
    plt.tight_layout()
    plt.show()

def analyze_taxa_differences(diagnosis_data, recurrence_dict, classification):
    """
    Performs statistical analysis to identify taxa that significantly differ
    between patients with BV recurrence and those without.

    Parameters:
    -----------
    diagnosis_data : DataFrame
        Dataset with microbiome information
    recurrence_dict : dict
        Dictionary mapping subjects to recurrence patterns
    classification : DataFrame
        Taxonomic classification information

    Returns:
    --------
    results_df_top : DataFrame
        Top OTUs by p-value (up to 10 rows)
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu

    # Extract baseline data (Follow-up = 1)
    baseline_data = diagnosis_data[diagnosis_data['Follow-up'] == '1'].copy()

    # Assign recurrence status using detailed pattern
    baseline_data['Has_Recurrence'] = baseline_data['Subject_ID'].map(
        {subject: 1 if info.get('type') in ['No_Initial_Response', 'Immediate_Recurrence',
                                            'Delayed_Recurrence', 'Any_Recurrence']
         else 0 for subject, info in recurrence_dict.items()}
    )

    # Drop rows without recurrence status
    baseline_data = baseline_data.dropna(subset=['Has_Recurrence'])

    # Get OTU columns
    otu_columns = [col for col in baseline_data.columns if col.startswith('otu')]

    # Calculate relative abundance (if not already normalized)
    if 'total_reads' in baseline_data.columns:
        for otu in otu_columns:
            baseline_data[f'rel_{otu}'] = baseline_data[otu].astype(float) / baseline_data['total_reads'].astype(float)
    else:
        for otu in otu_columns:
            baseline_data[f'rel_{otu}'] = baseline_data[otu].astype(float)  # assume normalized

    rel_otu_columns = [f'rel_{otu}' for otu in otu_columns]

    # Split groups
    recurrence_group = baseline_data[baseline_data['Has_Recurrence'] == 1]
    non_recurrence_group = baseline_data[baseline_data['Has_Recurrence'] == 0]

    print("Recurrence group size:", len(recurrence_group))
    print("Non-recurrence group size:", len(non_recurrence_group))

    if len(recurrence_group) == 0 or len(non_recurrence_group) == 0:
        print("Insufficient data: both recurrence and non-recurrence groups are required for comparison.")
        return pd.DataFrame()

    # Perform Mann-Whitney U tests
    results = []
    for rel_otu in rel_otu_columns:
        otu_name = rel_otu.replace('rel_', '')
        otu_num = otu_name.replace('otu', '')

        try:
            tax_class = classification[classification['#OTU'] == otu_num]['Taxonomic classification'].values[0]
        except (IndexError, KeyError):
            tax_class = 'Unknown'

        rec_values = recurrence_group[rel_otu].dropna()
        non_rec_values = non_recurrence_group[rel_otu].dropna()

        if len(rec_values) > 0 and len(non_rec_values) > 0:
            try:
                stat, p_value = mannwhitneyu(rec_values, non_rec_values, alternative='two-sided')
                mean_rec = rec_values.mean()
                mean_non_rec = non_rec_values.mean()
                fold_change = (mean_rec / mean_non_rec) if mean_non_rec > 0 else float('inf')

                results.append({
                    'OTU': otu_name,
                    'Taxonomy': tax_class,
                    'Mean_Recurrence': mean_rec,
                    'Mean_Non_Recurrence': mean_non_rec,
                    'Fold_Change': fold_change,
                    'p_value': p_value
                })
            except Exception as e:
                print(f"Stat test failed for {otu_name}: {e}")

    # Convert results
    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No valid comparisons could be made. Check OTU data.")
        return results_df

    results_df.sort_values(by='p_value', inplace=True)

    # Bonferroni correction
    results_df['p_adjusted'] = results_df['p_value'] * len(results_df)
    results_df['p_adjusted'] = results_df['p_adjusted'].clip(upper=1.0)


    # Plot significant taxa
    significant_taxa = results_df[results_df['p_adjusted'] < 0.05]
    if not significant_taxa.empty:
        top_n = min(15, len(significant_taxa))
        top_taxa = significant_taxa.head(top_n)

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_taxa)), top_taxa['Fold_Change'], align='center')
        plt.yticks(range(len(top_taxa)), [f"{row['OTU']} - {row['Taxonomy']}" for _, row in top_taxa.iterrows()])
        plt.axvline(x=1, color='red', linestyle='--')
        plt.xscale('log')
        plt.xlabel('Fold Change (Recurrence / Non-recurrence)', fontsize=12)
        plt.title('Significant Taxa Differentiating BV Recurrence', fontsize=14)
        plt.tight_layout()
        plt.show()
    else:
        print("No taxa showed significant differences after multiple testing correction.")

    return results_df.head(10)

def identify_early_warning_signs(diagnosis_data, recurrence_dict, classification):
    """
    Identifies microbiome characteristics during remission that predict subsequent recurrence.

    Parameters:
    -----------
    diagnosis_data : DataFrame
        Dataset with microbiome information
    recurrence_dict : dict
        Dictionary mapping subjects to recurrence patterns
    classification : DataFrame
        Taxonomic classification information

    Returns:
    --------
    early_warning_model : RandomForestClassifier or None
        Model trained to predict imminent recurrence, if enough data available
    """
    # Get subjects who had recurrence vs. successful treatment
    recurrence_subjects = [subject for subject, info in recurrence_dict.items()
                          if info.get('has_recurrence', False)]

    successful_subjects = [subject for subject, info in recurrence_dict.items()
                          if info.get('type') == 'Successful_Treatment']

    # Extract data from follow-ups where BV status was negative (remission)
    remission_data = diagnosis_data[diagnosis_data['BV_status'] == 0].copy()

    # Flag whether this remission was followed by recurrence
    remission_data['Pre_Recurrence'] = np.nan

    for idx, row in remission_data.iterrows():
        subject = row['Subject_ID']
        follow_up = row['Follow-up']

        # Check if this subject is in the recurrence group
        if subject in recurrence_subjects:
            # Check if there was a positive BV status in any later follow-up
            later_followups = diagnosis_data[(diagnosis_data['Subject_ID'] == subject) &
                                           (pd.to_numeric(diagnosis_data['Follow-up']) >
                                            pd.to_numeric(follow_up))]

            if any(later_followups['BV_status'] == 1):
                remission_data.loc[idx, 'Pre_Recurrence'] = 1
            else:
                remission_data.loc[idx, 'Pre_Recurrence'] = 0

        # If subject is in successful treatment group
        elif subject in successful_subjects:
            remission_data.loc[idx, 'Pre_Recurrence'] = 0

    # Drop rows with NaN in Pre_Recurrence
    remission_data = remission_data.dropna(subset=['Pre_Recurrence'])

    # If we don't have enough data, return early
    if len(remission_data) < 10:
        print("Not enough data for early warning signs analysis")
        return None

    # Get OTU columns
    otu_columns = [col for col in remission_data.columns if col.startswith('otu')]

    # Calculate relative abundance
    for otu in otu_columns:
        remission_data[f'rel_{otu}'] = remission_data[otu].astype(float) / remission_data['total_reads'].astype(float)

    rel_otu_columns = [f'rel_{otu}' for otu in otu_columns]

    # Compare microbiomes in remission before recurrence vs. successful
    recurrence_imminent = remission_data[remission_data['Pre_Recurrence'] == 1]
    recurrence_avoided = remission_data[remission_data['Pre_Recurrence'] == 0]

    if len(recurrence_imminent) < 3 or len(recurrence_avoided) < 3:
        print("Not enough samples in each group for meaningful comparison")
        return None

    # Calculate mean relative abundance for each group
    imminent_means = recurrence_imminent[rel_otu_columns].mean()
    avoided_means = recurrence_avoided[rel_otu_columns].mean()

    # Calculate fold difference
    fold_diff = imminent_means / (avoided_means + 0.0001)  # Add small constant to avoid division by zero
    fold_diff = fold_diff.sort_values(ascending=False)

    ## Mann - Whitney U-Test ##
    from scipy.stats import mannwhitneyu

    # Perform Mann-Whitney U test for each OTU
    results = []
    for otu in rel_otu_columns:
        # Extract OTU name without rel_ prefix
        otu_name = otu.replace('rel_', '')
        otu_num = otu_name.replace('otu', '')

        # Get taxonomic classification
        try:
            tax_class = classification[classification['#OTU'] == otu_num]['Taxonomic classification'].values[0]
        except (IndexError, KeyError):
            tax_class = 'Unknown'

        # Extract relative abundance values from remission data
        rec_values = recurrence_imminent[otu].dropna()
        non_rec_values = recurrence_avoided[otu].dropna()

        if len(rec_values) > 0 and len(non_rec_values) > 0:
            try:
                stat, p_value = mannwhitneyu(rec_values, non_rec_values, alternative='two-sided')

                mean_rec = rec_values.mean()
                mean_non_rec = non_rec_values.mean()

                # Avoid division by zero
                if mean_non_rec == 0:
                    fold_change = float('inf') if mean_rec > 0 else 0
                else:
                    fold_change = mean_rec / mean_non_rec

                results.append({
                    'OTU': otu_name,
                    'Taxonomy': tax_class,
                    'Mean_Recurrence': mean_rec,
                    'Mean_Non_Recurrence': mean_non_rec,
                    'Fold_Change': fold_change,
                    'p_value': p_value
                })
            except Exception as e:
                print(f"Statistical test failed for {otu_name}: {e}")

    results_df = pd.DataFrame(results)
    results_df.sort_values(by='p_value', inplace=True)
    results_df['p_adjusted'] = results_df['p_value'] * len(results_df)
    results_df['p_adjusted'] = results_df['p_adjusted'].clip(upper=1.0)

    display(results_df)

    # Plot top differentiating OTUs
    top_n = 10
    top_otus = fold_diff.head(top_n).index

    top_otus_cleaned = top_otus.str.replace('rel_', '').str.replace('otu', '')

    top_otus_with_taxonomy = get_otu_taxonomy_mapping(top_otus_cleaned, classification)


    # plt.figure(figsize=(12, 6))
    # plt.bar(top_otus_with_taxonomy, fold_diff[top_otus])
    # plt.axhline(y=1, color='r', linestyle='--')
    # plt.title('Top OTUs Enriched in Remission Before Recurrence', fontsize=14)
    # plt.ylabel('Fold Difference (Pre-Recurrence / No Recurrence)', fontsize=12)
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()

    # Build new Series: index = taxonomic labels, values = fold differences
    fold_diff_tax_labeled = pd.Series(
        data=fold_diff[top_otus].values,
        index=top_otus_with_taxonomy
    )

    print("Top OTUs Enriched in Remission Before Recurrence:")
    print(fold_diff_tax_labeled)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(fold_diff_tax_labeled.index, fold_diff_tax_labeled.values)
    plt.axhline(y=1, color='r', linestyle='--')
    plt.title('Top OTUs Enriched in Remission Before Recurrence', fontsize=14)
    plt.ylabel('Fold Difference (Pre-Recurrence / No Recurrence)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Build a model to predict imminent recurrence based on remission microbiome
    X = remission_data[otu_columns]
    y = remission_data['Pre_Recurrence']

    # Check if we have both classes represented
    if len(np.unique(y)) < 2:
        print("Only one class present in data - cannot train model")
        return None

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Create and train the model
    early_warning_model = RandomForestClassifier(n_estimators=100, random_state=42)
    early_warning_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = early_warning_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nImminent BV Recurrence Model Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))


    # Create and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Recurrence', 'Recurrence'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix for Imminent BV Recurrence Prediction')
    plt.show()

    # ROC curve
    y_pred_proba = early_warning_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Imminent BV Recurrence Prediction')
    plt.legend(loc="lower right")
    plt.show()

    # Feature importance
    importances = early_warning_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Get taxonomic information for top OTUs
    top_n = 15  # Show top 15 features
    top_otus = [otu_columns[i] for i in indices[:top_n]]

    # Create labels combining OTU and taxonomic name
    top_otus_with_taxonomy = get_otu_taxonomy_mapping(top_otus, classification)

    # Plot feature importance with taxonomic names
    plt.figure(figsize=(12, 8))
    plt.title("Top Taxa Predicting Imminent BV Recurrence", fontsize=16)
    plt.barh(range(len(top_otus_with_taxonomy)), importances[indices[:len(top_otus_with_taxonomy)]], align="center")
    plt.yticks(range(len(top_otus_with_taxonomy)), top_otus_with_taxonomy)
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.show()

    return early_warning_model

def get_otu_taxonomy_mapping(otu_list, classification):
    """
    Creates labels that combine OTU IDs with their taxonomic classification.

    Parameters:
    -----------
    otu_list : list
        List of OTU IDs (e.g., 'otu1', 'otu2')
    classification : DataFrame
        DataFrame containing taxonomic classifications

    Returns:
    --------
    labels : list
        List of combined OTU and taxonomy labels
    """
    labels = []
    for otu in otu_list:
        otu_num = otu.replace('otu', '')
        try:
            tax_info = classification[classification['#OTU'] == otu_num]['Taxonomic classification'].values[0]
            labels.append(f"{otu} - {tax_info}")
        except (IndexError, KeyError):
            labels.append(f"{otu} - Unknown")
    return labels
# Example run

diagnosis_data, recurrence_dict = label_comprehensive_recurrence_patterns(diagnosis_data)
visualize_recurrence_patterns(recurrence_dict)

diagnosis_data_norm = normalize_otus(diagnosis_data, OTU_names)

diagnosis_data_norm['Follow-up'] = diagnosis_data['Follow-up']
diagnosis_data_norm['Subject_ID'] = diagnosis_data['Subject_ID']
diagnosis_data_norm['total_reads'] = diagnosis_data['total_reads']
diagnosis_data_norm['BV_status'] = diagnosis_data['BV_status']
diagnosis_data_norm['Sample_ID'] = diagnosis_data['Sample_ID']
diagnosis_data_norm['Amsel_discharge'] = diagnosis_data['Amsel_discharge']
diagnosis_data_norm['Amsel_odor'] = diagnosis_data['Amsel_odor']
diagnosis_data_norm['Amsel_clue'] = diagnosis_data['Amsel_clue']
diagnosis_data_norm['pH'] = diagnosis_data['pH']
diagnosis_data_norm['Amsel_score'] = diagnosis_data['Amsel_score']
diagnosis_data_norm['Nugent_Score'] = diagnosis_data['Nugent_Score']

plot_pca_microbiome(diagnosis_data_norm, recurrence_dict)

analyze_taxa_differences(diagnosis_data, recurrence_dict, classification)

identify_early_warning_signs(diagnosis_data, recurrence_dict, classification)
