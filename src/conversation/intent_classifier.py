# Extracted Code: Intent Classification System
# This section contains the code for preparing features, addressing class imbalance with SMOTE, and training/tuning various classification models for news article intent classification.

# Prepare features for classification
print("🔧 Preparing features for classification...")

# 💡 TIP: Combine multiple feature types for better performance
# - TF-IDF features (most important)
# - Sentiment features
# - Text length features
# - POS features (if available)

# Create feature matrix
X_tfidf = tfidf_matrix.toarray()  # TF-IDF features

# Add sentiment features
sentiment_features = sentiment_df[['full_sentiment', 'pos_score', 'neu_score', 'neg_score']].values

# Add text length features
length_features = np.array([
    df['full_text'].str.len(),  # Character length
    df['full_text'].str.split().str.len(),  # Word count
    df['headline'].str.len(),  # Title length
]).T

# Add POS features
# Ensure pos_df is aligned with df based on article_id
pos_features_aligned = df[['article_id']].merge(pos_df, on='article_id', how='left').drop(columns=['article_id', 'category']).fillna(0).values

# 🚀 YOUR CODE HERE: Combine all features
X_combined = np.hstack([
    X_tfidf,
    sentiment_features,
    length_features,
    pos_features_aligned
])

# Target variable
y = df['category'].values

print(f"✅ Feature matrix prepared!")
print(f"📊 Feature matrix shape: {X_combined.shape}")
print(f"🎯 Number of classes: {len(np.unique(y))}")
print(f"📋 Classes: {np.unique(y)}")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📈 Data split:")
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")

from imblearn.over_sampling import SMOTE

print("🧪 Applying SMOTE to training data...")

# Initialize SMOTE
smote = SMOTE(random_state=42, sampling_strategy='auto') # 'auto' balances all minority classes

# Apply SMOTE to the training data
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print(f"✅ SMOTE application complete!")
print(f"📈 Original training samples: {X_train.shape[0]}")
print(f"📊 Resampled training samples: {X_resampled.shape[0]}")
print("Distribution of classes after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Train and evaluate multiple classifiers
print("🤖 Training multiple classifiers...")

# Define classifiers to compare
classifiers = {
    'Naive Bayes': MultinomialNB(),
    # Reduced max_iter for faster training and set solver for efficiency with this dataset size
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=400, solver='liblinear', class_weight='balanced'),
    # Optimized SVM for quicker training: using LinearSVC for linear kernel.
    # LinearSVC is based on liblinear, which is more efficient for large datasets
    # compared to SVC with a linear kernel. It does not support 'probability=True'.
    'SVM': LinearSVC(random_state=42, dual=False, max_iter=1000, class_weight='balanced') # dual=False recommended for n_samples > n_features
}

# 💡 TIP: For larger datasets, you might want to use SGDClassifier for efficiency
# from sklearn.linear_model import SGDClassifier
# classifiers['SGD'] = SGDClassifier(random_state=42)

# Train and evaluate each classifier
results = {}
trained_models = {}

# Identify the index of the 'full_sentiment' column to exclude it for MultinomialNB
# X_tfidf.shape[1] is the number of TF-IDF features
# Then comes full_sentiment (index 0 of sentiment_features), pos_score, neu_score, neg_score
sentiment_start_idx = X_tfidf.shape[1]
full_sentiment_idx = sentiment_start_idx

# Create a feature set for MultinomialNB that excludes the 'full_sentiment' column
# This means taking all TF-IDF features, and then sentiment features from pos_score onwards,
# and then length features.
X_train_multinomial = np.hstack([
    X_resampled[:, :sentiment_start_idx],  # All TF-IDF features
    X_resampled[:, (full_sentiment_idx + 1):] # Skip full_sentiment, take other sentiment and length features
])
X_test_multinomial = np.hstack([
    X_test[:, :sentiment_start_idx],  # All TF-IDF features
    X_test[:, (full_sentiment_idx + 1):] # Skip full_sentiment, take other sentiment and length features
])


for name, classifier in classifiers.items():
    print(f"\n🔄 Training {name}...")

    # 🚀 YOUR CODE HERE: Train and evaluate classifier
    if name == 'Naive Bayes':
        X_train_clf = X_train_multinomial
        X_test_clf = X_test_multinomial
    else:
        X_train_clf = X_resampled # Use resampled data
        X_test_clf = X_test

    # Train the model
    classifier.fit(X_train_clf, y_resampled) # Use resampled labels

    # Make predictions
    y_pred = classifier.predict(X_test_clf)
    y_pred_proba = classifier.predict_proba(X_test_clf) if hasattr(classifier, 'predict_proba') else None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Cross-validation score
    cv_scores = cross_val_score(classifier, X_train_clf, y_resampled, cv=3, scoring='accuracy') # Use resampled labels for CV

    # Store results
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

    trained_models[name] = classifier

    print(f"  ✅ Accuracy: {accuracy:.4f}")
    print(f"  📊 CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print("\n🏆 CLASSIFIER COMPARISON")
print("=" * 50)
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Test Accuracy': [results[name]['accuracy'] for name in results.keys()],
    'CV Mean': [results[name]['cv_mean'] for name in results.keys()],
    'CV Std': [results[name]['cv_std'] for name in results.keys()]
})

print(comparison_df.round(4))

# Find best model
best_model_name = comparison_df.loc[comparison_df['Test Accuracy'].idxmax(), 'Model']
print(f"\n🥇 Best performing model: {best_model_name}")


from sklearn.model_selection import GridSearchCV

print("⚙️ Starting GridSearchCV for SVM...")

# Define the parameter grid for LinearSVC
# We'll tune the 'C' parameter (regularization strength)
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'loss': ['hinge', 'squared_hinge'] # Different loss functions for LinearSVC
}

# Initialize LinearSVC with balanced class weights, as determined previously
svm_tuned = LinearSVC(random_state=42, dual=False, max_iter=1000, class_weight='balanced')

# Setup GridSearchCV
# We'll use 3-fold cross-validation and accuracy as the scoring metric
grid_search = GridSearchCV(estimator=svm_tuned, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

print("✅ GridSearchCV for SVM complete!")

# Print the best parameters and best score
print(f"\n🥇 Best parameters for SVM: {grid_search.best_params_}")
print(f"📈 Best cross-validation score (accuracy): {grid_search.best_score_:.4f}")

# Update the best model with the tuned SVM
best_model_name_tuned = 'SVM (Tuned)'
trained_models[best_model_name_tuned] = grid_search.best_estimator_
results[best_model_name_tuned] = {
    'accuracy': grid_search.score(X_test, y_test), # Evaluate on test set
    'cv_mean': grid_search.best_score_,
    'cv_std': grid_search.cv_results_['std_test_score'][grid_search.best_index_],
    'predictions': grid_search.best_estimator_.predict(X_test),
    'probabilities': None # LinearSVC does not support predict_proba
}

# Update best_model_name if tuned SVM is better
if results[best_model_name_tuned]['accuracy'] > results[best_model_name]['accuracy']:
    best_model_name = best_model_name_tuned
    print(f"\n🏆 Tuned SVM is the new best model with test accuracy: {results[best_model_name_tuned]['accuracy']:.4f}")
else:
    print(f"\nℹ️ Original SVM model ({best_model_name}) remains the best with test accuracy: {results[best_model_name]['accuracy']:.4f}")


print("⚙️ Starting GridSearchCV for Logistic Regression...")

# Define the parameter grid for Logistic Regression
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear'] # Keep liblinear as it's efficient for this type of data
}

# Initialize Logistic Regression with balanced class weights
lr_tuned = LogisticRegression(random_state=42, max_iter=400, class_weight='balanced')

# Setup GridSearchCV
grid_search_lr = GridSearchCV(estimator=lr_tuned, param_grid=param_grid_lr,
                               cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit GridSearchCV to the resampled training data
grid_search_lr.fit(X_resampled, y_resampled)

print("✅ GridSearchCV for Logistic Regression complete!")

# Print the best parameters and best score
print(f"\n🥇 Best parameters for Logistic Regression: {grid_search_lr.best_params_}")
print(f"📈 Best cross-validation score (accuracy): {grid_search_lr.best_score_:.4f}")

# Update the best model with the tuned Logistic Regression
best_model_name_tuned_lr = 'Logistic Regression (Tuned)'
trained_models[best_model_name_tuned_lr] = grid_search_lr.best_estimator_
results[best_model_name_tuned_lr] = {
    'accuracy': grid_search_lr.score(X_test, y_test), # Evaluate on test set
    'cv_mean': grid_search_lr.best_score_,
    'cv_std': grid_search_lr.cv_results_['std_test_score'][grid_search_lr.best_index_],
    'predictions': grid_search_lr.best_estimator_.predict(X_test),
    'probabilities': grid_search_lr.best_estimator_.predict_proba(X_test)
}

# Update best_model_name if tuned Logistic Regression is better
if results[best_model_name_tuned_lr]['accuracy'] > results[best_model_name]['accuracy']:
    best_model_name = best_model_name_tuned_lr
    print(f"\n🏆 Tuned Logistic Regression is the new best model with test accuracy: {results[best_model_name_tuned_lr]['accuracy']:.4f}")
else:
    print(f"\nℹ️ Current best model ({best_model_name}) remains the best with test accuracy: {results[best_model_name]['accuracy']:.4f}")
