import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv("HybridDataset.csv")


# Preprocessing/Cleaning Data
df_clean = df.copy()
for col in df_clean.columns:
   if df_clean[col].dtype == 'object':
       df_clean[col] = df_clean[col].str.strip().str.lower()


df_clean = df_clean.rename(columns={
   'what is your age group?': 'age_group',
   'what is your occupation?': 'occupation',
   'how many hours per day do you spend online?': 'hours_per_day',
   'what device do you use most to access the internet?': 'device_used',
   'what is the total number of social media platforms that you use?': 'num_platforms',
   'what is your most used social media platform?': 'most_used_platform',
   'could you spend an entire week without social media?': 'week_without_sm',
   'the first thing you do after walking up is scroll through your social media account.': 'sm_first_thing',
   'how often do you find yourself distracted while working or studying due to social media?': 'distraction_freq'
})


df_clean['num_platforms_numeric'] = df_clean['num_platforms'].replace({'4 +': 4, '3': 3, '2': 2, '1': 1}).astype(float)
df_clean['hours_per_day'] = df_clean['hours_per_day'].replace({
   'less than 2': '<2', '2-4': '2-4', '4-6': '4-6', '6-8': '6-8', '8+': '>8'
}).str.replace('"', '').str.strip().str.lower()


#Graph 1: SM Platforms vs Occupation
sns.countplot(data=df_clean, x='num_platforms', hue='occupation')
plt.title('Number of SM Platforms Used by Occupation')
plt.tight_layout()
plt.show()


#Graph 2: First Thing in Morning vs Age Group
sns.countplot(data=df_clean, x='age_group', hue='sm_first_thing')
plt.title('SM First Thing in Morning by Age Group')
plt.tight_layout()
plt.show()


#Graph 3: Hours Online vs SM Platforms (Boxplot)
order_hours = ['<2', '2-4', '4-6', '6-8', '>8']
sns.boxplot(data=df_clean, x='hours_per_day', y='num_platforms_numeric', order=order_hours)
plt.title('SM Platform Count vs. Hours Online')
plt.tight_layout()
plt.show()


#Graph 4: Most Used Platform vs SM Platforms
sns.boxplot(data=df_clean, x='most_used_platform', y='num_platforms_numeric')
plt.title('Most Used Platform vs. Number of SM Platforms')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#Graph 5: Distraction Frequency vs SM Platform Count
sns.countplot(data=df_clean, x='distraction_freq', hue='num_platforms')
plt.title('Distraction Frequency vs. SM Platform Count')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


#Graph 6: Stacked Bar - First Thing in Morning by Age
pd.crosstab(df_clean['age_group'], df_clean['sm_first_thing']).plot(kind='bar', stacked=True)
plt.title('SM First Thing in Morning (Stacked) by Age Group')
plt.tight_layout()
plt.show()


#Graph 7: Correlation Heatmap
df_clean['distraction_score'] = df_clean['distraction_freq'].map({
   'never': 0, 'rarely': 1, 'sometimes': 2, 'often': 3, 'very often': 4
})
df_clean['first_thing_score'] = df_clean['sm_first_thing'].map({
   'no': 0, 'yes, even though i know that\'s a bad routine.': 1
})
sns.heatmap(df_clean[['num_platforms_numeric', 'distraction_score', 'first_thing_score']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap: SM Usage & Distraction')
plt.tight_layout()
plt.show()


#Graph 8: SM Use During Work vs SM Platforms (Grouped Bar)
sns.countplot(data=df_clean, x='num_platforms', hue='distraction_freq')
plt.title('SM Use During Work vs. Number of Platforms')
plt.tight_layout()
plt.show()

#Simple Regression and Linear Regression (multiple)
#now that we've finished the EDA and preprocessing portion of the code, now we are
#moving on to the regression portion of the code
#using regression allows us to see which social media usage behaviors like screen time, platform count, morning scrolling, are
#ones that are most predictive of distractions




df = pd.read_csv("HybridDataset.csv")
#cleaning up our df to get rid of mismatched values for hours
df = df.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)
#this is the dependent variable, we are using for distraction
df['distraction_score'] = df['how often do you find yourself distracted while working or studying due to social media?'].map({
   'never': 0, 'rarely': 1, 'sometimes': 2, 'often': 3, 'very often': 4
})
# number of platforms score value >> independent variable + a predictor
df['num_platforms_numeric'] = df['what is the total number of social media platforms that you use?'].map({
   '1': 1, '2': 2, '3': 3, '4 +': 4
})
#hours online, again a predictor value
df['how many hours per day do you spend online?'] = df['how many hours per day do you spend online?']\
   .str.strip().str.lower().str.replace('"', '')
#making dictionary for the ranges of hours
hours_map = {
   'less than 2': 1,
   '2-4': 2,
   '4-6': 3,
   '6-8': 4,
   'more than 8': 5
}
df['hours_score'] = df['how many hours per day do you spend online?'].map(hours_map)


# predictor value, is it the first thing in the morning / yes/no value
df['first_thing_score'] = df['the first thing you do after walking up is scroll through your social media account.'].map({
   'no': 0,
   "yes, even though i know that's a bad routine.": 1
})
# purchasing premium subscriptions, again yes/no values, but is a predcitor
df['premium_score'] = df['do you purchase premium subscriptions for social media platforms? (like spotify or netflix premium)'].map({
   'yes': 1, 'no': 0
})
# cleaning up via dropping missing rows
df_model = df.dropna(subset=[
   'num_platforms_numeric', 'hours_score', 'first_thing_score',
   'premium_score', 'distraction_score', 'what is your age group?'
])
#hot encoding age group
df_model = pd.get_dummies(df_model, columns=['what is your age group?'], drop_first=True)


# simple regression >> this means we are only using one predictor value to determine r^2
X_simple = df_model[['num_platforms_numeric']]
y = df_model['distraction_score']
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)


model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)
y_pred_s = model_simple.predict(X_test_s)


print("\n Simple Regression ")
print("R² Value:", r2_score(y_test_s, y_pred_s))
print("MSE Value:", mean_squared_error(y_test_s, y_pred_s))




# now we are doing multiple regression, using multiple predictor values for r^2
features = ['num_platforms_numeric', 'hours_score', 'first_thing_score', 'premium_score'] + \
          [col for col in df_model.columns if col.startswith('what is your age group?_')]


X_multi = df_model[features]
y = df_model['distraction_score']


X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)


model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)
y_pred_m = model_multi.predict(X_test_m)


print("\n Multiple Regression ")
print("R² Value:", r2_score(y_test_m, y_pred_m))
print("MSE Value:", mean_squared_error(y_test_m, y_pred_m))


# printing our coefficient table
coef_df = pd.DataFrame({
   'Feature': X_multi.columns,
   'Coefficient': model_multi.coef_
})
print("\nCoefficients Table:")
print(coef_df)




#plotting the relationships, coefficient tables, and predictors
import matplotlib.pyplot as plt


# R^2 scores plots
r2_scores = [-0.0225, 0.089]  #our values that our r^2 returns
labels = ['Simple Regression', 'Multiple Regression']


plt.bar(labels, r2_scores, color=['skyblue', 'purple'])
plt.ylabel('R² Score')
plt.title('Comparison of Model Performance')
plt.ylim(-0.05, 0.2)
plt.axhline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.show()




#continuous predictors graphs
import seaborn as sns


# Scatter + regression line for each predictor
predictors = ['num_platforms_numeric', 'hours_score', 'first_thing_score']
for predictor in predictors:
   sns.lmplot(data=df_model, x=predictor, y='distraction_score', aspect=1.5, scatter_kws={'alpha':0.4})
   plt.title(f'Distraction Score vs {predictor}')
   plt.tight_layout()
   plt.show()


#  catgorical predictors score, premium + age group
df_plot = df_model.copy()


# one-hot
age_cols = [col for col in df_plot.columns if col.startswith('what is your age group?_')]
df_plot['age_group'] = df_plot[age_cols].idxmax(axis=1).str.replace('what is your age group?_', '', regex=False)


# box plot for premium subscription
sns.boxplot(data=df_plot, x='premium_score', y='distraction_score')
plt.xticks([0, 1], ['No', 'Yes'])
plt.title('Distraction Score vs Premium Subscription')
plt.tight_layout()
plt.show()


# box plot for age group
sns.boxplot(data=df_plot, x='age_group', y='distraction_score')
plt.xticks(rotation=30)
plt.title('Distraction Score vs Age Group')
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Normalization BEFORE clustering
features_to_normalize = ['hours_score', 'num_platforms_numeric', 'first_thing_score', 'premium_score']
df_to_scale = df_model[features_to_normalize]
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df_to_scale)
df_normalized = pd.DataFrame(normalized_data, columns=features_to_normalize)

# Selecting the features to cluster based on the numeric features in the dataset
# Standardizing for the Kmeans and then applying that scale on the datacluster_features_scaled = df_normalized
wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(cluster_features_scaled)
    wcss.append(kmeans.inertia_)

# Implementing the Elbow Method to find ideal number of clusters
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o')
plt.title('Elbow Method - Choose Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Setting the k value to 3 because this is when there is the sharpest drop in WCSS from k =1 to k = 3
optimal_k = 3

# Apply the optimal number of cluster to the K-means
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(cluster_features_scaled)

# Adding cluster labels to dataset
df_model_clustered = df_model.copy()
df_model_clustered['cluster'] = cluster_labels

# Random Forest Regression
features_with_cluster = features + ['cluster']
X_clustered = df_model_clustered[features_with_cluster]
y_clustered = df_model_clustered['distraction_score']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clustered, y_clustered, test_size=0.2, random_state=42)

# Enhancing random forest model using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid,
                           scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search.fit(X_train_c, y_train_c)

# Finding the best model
best_rf_clustered = grid_search.best_estimator_
y_pred_clustered = best_rf_clustered.predict(X_test_c)
r2 = r2_score(y_test_c, y_pred_clustered)
mse = mean_squared_error(y_test_c, y_pred_clustered)

# Listing the feature importances with user distraction
importances = best_rf_clustered.feature_importances_
importances_df = pd.DataFrame({
    'Feature': X_clustered.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

# Printing results
print("\nRandom Forest Regression with KMeans Cluster Feature")
print(f"R² Score: {r2}")
print(f"MSE: {mse}\n")
print("Feature Importances:")
print(importances_df.to_string(index=False))

# Plotting feature importances to visually see which features have the most impact on user distraction
plt.figure(figsize=(10, 6))
sns.barplot(data=importances_df, x='Importance', y='Feature', palette='coolwarm')
plt.title('Feature Importances: Predicting Distraction from Social Media Behavior')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()