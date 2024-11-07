import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import f_oneway

# Suppressing warnings
import warnings
warnings.filterwarnings('ignore')

# Setting up visualization style
sns.set_style("whitegrid")
sns.set_palette("RdBu")

# Loading data
data = 'D:/TS2VG/all_fea_ex_v3.csv'
df = pd.read_csv(data)
# df['File'] = df['File'].apply(lambda x: 0 if 'ex01' in x else (1 if 'ex05' in x or 'ex06' in x or 'ex07' in x else x))
df['File'] = df['File'].apply(lambda x: 0 if 'ex05' in x else (1 if 'ex06' in x else None))
df.dropna(subset=['File'], inplace=True)
df.drop(['Fit', 'Channel'], axis=1, inplace=True)

# Initialize empty DataFrames to store results
anova_results = pd.DataFrame()

# Perform ANOVA only for versions 2, 3, 4, and 5
for version in range(2, 6):
    if version in [2, 3, 4, 5]:
        # Filtering data
        df1 = df[df['version'] == version]
        df1.drop('version', axis=1, inplace=True)

        # ANOVA Test
        features = df1.drop(['File', 'Nodes'], axis=1)
        target = df1['File']
        p_values = []
        feature_names = []
        for column in features.columns:
            statistic, p_value = f_oneway(features[column], target)
            feature_names.append(column)
            p_values.append(p_value)

        data1_anova = {'Feature': feature_names, 'p_value': p_values}
        df1_anova = pd.DataFrame(data1_anova)
        df1_anova['Version'] = version
        anova_results = pd.concat([anova_results, df1_anova], ignore_index=True)

# Save ANOVA results to a single CSV file
anova_results.to_csv('anova_test_v4.csv', index=False)

# Initialize empty DataFrames to store SVM and KNN results
svm_results = pd.DataFrame()
knn_results = pd.DataFrame()

# Loop through different versions and nodes
for version in range(2, 6):
    for nodes in [512, 1024, 2048, 4096]:
        print(f"Version: {version}, Nodes: {nodes}")

        # Filtering data
        df1 = df[df['version'] == version]
        df1 = df1[df1['Nodes'] == nodes]
        df1.drop('version', axis=1, inplace=True)

        # ANOVA feature selection
        columns_to_drop = anova_results[(anova_results['Version'] == version) & (anova_results['p_value'] > 0.0001)]['Feature'].tolist()
        df1.drop(columns=columns_to_drop, inplace=True)

        # Splitting data into train and test sets
        x = df1.drop(['File', 'Nodes'], axis=1)
        y = df1['File']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        # Standardizing features
        cols = x_train.columns
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        x_train = pd.DataFrame(x_train, columns=[cols])
        x_test = pd.DataFrame(x_test, columns=[cols])

        # SVM Classifier
        kernel_types = ['rbf', 'linear', 'poly', 'sigmoid']
        C_values = [1.0, 100.0, 1000.0]
        best_accuracy = 0
        best_kernel = None
        best_C = None
        best_y_pred = None

        for kernel in kernel_types:
            for C in C_values:
                svc = SVC(kernel=kernel, C=C)
                svc.fit(x_train, y_train)
                y_pred = svc.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_kernel = kernel
                    best_C = C
                    best_y_pred = y_pred

        best_svc = SVC(kernel=best_kernel, C=best_C)
        best_svc.fit(x_train, y_train)
        y_pred = best_svc.predict(x_test)

        cm = confusion_matrix(y_test, y_pred)
        TP = cm[0, 0]
        FN = cm[1, 0]
        FP = cm[0, 1]
        TN = cm[1, 1]
        precision = TP / float(TP + FP)
        recall = TP / float(TP + FN)
        specificity = TN / float(TN + FP)

        scores = cross_val_score(best_svc, x_train, y_train, cv=10, scoring='accuracy')
        svc = SVC()
        parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                      {'C': [1, 10, 100, 1000], 'kernel': ['rbf'],
                       'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
                      {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4],
                       'gamma': [0.01, 0.02, 0.03, 0.04, 0.05]},
                      {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5]}
                      ]
        grid_search = GridSearchCV(estimator=svc, param_grid=parameters, scoring='accuracy', cv=10, verbose=0)
        grid_search.fit(x_train, y_train)
        best_score = grid_search.best_score_

        svm_result = pd.DataFrame({'Version': version,
                                   'Nodes': nodes,
                                   'Best accuracy': best_accuracy,
                                   'Recall': recall,
                                   'Specificity': specificity,
                                   'Precision': precision,
                                   'Best score': best_score}, index=[0])
        svm_results = pd.concat([svm_results, svm_result], ignore_index=True)

        # KNN Classifier
        best_accuracy = 0
        best_k = None
        for k in range(3, 11):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train, y_train)
            y_pred = knn.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k

        knn_best = KNeighborsClassifier(n_neighbors=best_k)
        knn_best.fit(x_train, y_train)
        y_pred = knn_best.predict(x_test)

        cm_best = confusion_matrix(y_test, y_pred)
        TP = cm_best[0, 0]
        FN = cm_best[1, 0]
        FP = cm_best[0, 1]
        TN = cm_best[1, 1]
        precision = TP / float(TP + FP)
        recall = TP / float(TP + FN)
        specificity = TN / float(TN + FP)

        scores = cross_val_score(knn_best, x_train, y_train, cv=10, scoring='accuracy')
        scores_mean = scores.mean()

        knn_result = pd.DataFrame({'Version': version,
                                   'Nodes': nodes,
                                   'Best accuracy': best_accuracy,
                                   'Recall': recall,
                                   'Specificity': specificity,
                                   'Precision': precision,
                                   'Cross-validation mean score': scores_mean}, index=[0])
        knn_results = pd.concat([knn_results, knn_result], ignore_index=True)

# Format results in SVM DataFrame
svm_results['Best accuracy'] = svm_results['Best accuracy'].apply(lambda x: round(x, 4))
svm_results['Recall'] = svm_results['Recall'].apply(lambda x: round(x, 4))
svm_results['Specificity'] = svm_results['Specificity'].apply(lambda x: round(x, 4))
svm_results['Precision'] = svm_results['Precision'].apply(lambda x: round(x, 4))
svm_results['Best score'] = svm_results['Best score'].apply(lambda x: round(x, 4))

# Save SVM results to CSV
svm_results.to_csv('svm_v4.csv', index=False)

# Format results in KNN DataFrame
knn_results['Best accuracy'] = knn_results['Best accuracy'].apply(lambda x: round(x, 4))
knn_results['Recall'] = knn_results['Recall'].apply(lambda x: round(x, 4))
knn_results['Specificity'] = knn_results['Specificity'].apply(lambda x: round(x, 4))
knn_results['Precision'] = knn_results['Precision'].apply(lambda x: round(x, 4))
knn_results['Cross-validation mean score'] = knn_results['Cross-validation mean score'].apply(lambda x: round(x, 4))

# Save KNN results to CSV
knn_results.to_csv('knn_v4.csv', index=False)
