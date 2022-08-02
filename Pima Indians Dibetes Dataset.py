import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import probplot
from scipy.stats import boxcox
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


def Read_csv(path):
    Data = pd.read_csv(path)
    return Data


# Convert Datatypes
def Data_types(dict_datatype, Data):
    Data = Data.astype(dict_datatype)
    return Data


# See Distribution of Numerical Variables using probability density function
def Distribution_plot(Data):
    Numeric = Data.select_dtypes(include='number')
    for i in Numeric:
        sns.histplot(Data, x=i, kde=True)
        plt.show()
    # for j in Data:
    #     sns.countplot(x=j, hue='Outcome', data=Data)
    #     plt.show()


# See Distribution of Numerical Variables using probability density function
def Distribution_QQ_plot(Data):
    Numeric = Data.select_dtypes(include='number')
    for i in Numeric:
        probplot(Data[i], dist="norm", plot=plt)
        plt.ylabel(i)
        plt.show()


# See Boxplot for Outliers
def Box_plot(Data):
    Numeric = Data.select_dtypes(include='number')
    for i in Numeric:
        Numeric.boxplot(column=i)
        plt.show()


# Test to validate normal distribution
def Gaussian_test(Data):
    is_normalised = pd.DataFrame(columns=['Variable', 'pvalue', 'pvalue1', 'Is_Normalised'])
    Numeric = Data.select_dtypes(include='number')
    for i in Numeric:
        alpha = 0.05
        statistics, p = shapiro(Data[i])
        Statistics1, p1 = normaltest(Data[i])
        if p > alpha or p1 > alpha:
            result = True
        else:
            result = False
        is_normalised = is_normalised.append(
            {'Variable': i, 'pvalue': p, 'pvalue1': p1, 'Is_Normalised': result}, ignore_index=True)
    return is_normalised


# Found outliers using IQR
def Outliers(Data):
    Dict = {}
    for i in Data:
        if Data[i].dtype.name == 'float64' or Data[i].dtype.name == 'int64' or Data[i].dtype.name == 'complex':
            Q1, Q3 = np.percentile(Data[i], [25, 75])
            iqr_val = Q3 - Q1
            lbv = Q1 - (1.5 * iqr_val)
            hbv = Q3 + (1.5 * iqr_val)
            temp_list = []
            for j in Data[i].values:
                if (j > hbv) or (j < lbv):
                    temp_list.append(j)
                    Dict.update({i: temp_list})

    return Dict


# Replace outliers
def Replace_outliers(Data):
    for i in Data:
        if Data[i].dtype.name == 'float64' or Data[i].dtype.name == 'int64' or Data[i].dtype.name == 'complex':
            Q1, Q3 = np.percentile(Data[i], [25, 75])
            iqr_val = Q3 - Q1
            lbv = Q1 - (1.5 * iqr_val)
            hbv = Q3 + (1.5 * iqr_val)
            Data.loc[Data[i] >= hbv, i] = hbv
            Data.loc[Data[i] <= lbv, i] = hbv
    return Data


# Scale Data using Standard_Scaler

def S_scaler(Data):
    Columns = Data.iloc[:, :-1]
    scaler = ColumnTransformer([('Standard', StandardScaler(), Columns.columns)], remainder='passthrough')
    data_scaled = scaler.fit_transform(Data)
    data_scaled = pd.DataFrame(data_scaled, index=Data.index, columns=Data.columns)
    return data_scaled


# Scale Data using Normalization

def N_scaler(Data):
    Columns = Data.iloc[:, :-1]
    scaler = ColumnTransformer([('Standard', MinMaxScaler(), Columns.columns)], remainder='passthrough')
    data_scaled = scaler.fit_transform(Data)
    data_scaled = pd.DataFrame(data_scaled, index=Data.index, columns=Data.columns)
    return data_scaled


# Correlation between Variables Plot

def Cor_plot(Data):
    Numeric = Data.select_dtypes(include='number')
    plt.figure(figsize=(12, 10))
    sns.pairplot(Numeric)
    plt.show()
    cor = Data.corr(method='spearman')
    sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
    plt.show()


# Correlation between Variables Quantify
def Correlation_cols(Data, Threshold):
    Correlated = []
    Cor = Data.corr(method='spearman')
    for indices, row in Cor.iterrows():
        for i in range(len(row)):
            if row[i] > Threshold:
                if row.index[i] != indices:
                    Correlated.append([row[i], row.index[i], indices])
    return Correlated


# Gaussian Transformation
def Gaussian_transform(Data, Transformer):
    Transform = Transformer
    Transformed_data = Data.copy()
    Gaussian_test_1 = Gaussian_test(Data)
    if Transform == 'Log':
        for values in Gaussian_test_1['Variable']:
            for column in Transformed_data:
                if column == values:
                    Transformed_data[column] = np.log(Transformed_data[column] + 1)
                # Distribution_plot(Data)
    elif Transform == 'Reciprocal':
        for values in Gaussian_test_1['Variable']:
            for column in Transformed_data:
                if column == values:
                    Transformed_data[column] = 1 / (Transformed_data[column] + 1)
    elif Transform == 'Square root':
        for values in Gaussian_test_1['Variable']:
            for column in Transformed_data:
                if column == values:
                    Transformed_data[column] = Transformed_data[column] ** (1 / 2)
    elif Transform == 'Exponential':
        for values in Gaussian_test_1['Variable']:
            for column in Transformed_data:
                if column == values:
                    Transformed_data[column] = Transformed_data[column] ** (1 / 5)
    elif Transform == 'Box Cox':
        for values in Gaussian_test_1['Variable']:
            for column in Transformed_data:
                if column == values:
                    Transformed_data[column], param = boxcox(Transformed_data[column] + 1)
    Distribution_plot(Transformed_data)
    print(Gaussian_test(Transformed_data))
    return Transformed_data


# Train test split
def X_Y_drop(Data, Output, size):
    x = Data.drop(Output, axis=1)
    y = Data[Output]
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=size, random_state=1)
    return X_train, X_test, Y_train, Y_test


# Logistic Regression

def Log_regression(x, y, X, Y):
    model = LogisticRegression()
    model.fit(x, y)
    y_pred = model.predict(X)
    y_pred_prob = model.predict_proba(X_test)
    print('Accuracy :' + str(model.score(X, Y)))
    print("Confusion Metric :" + str(confusion_matrix(Y, y_pred)))
    print("LR test roc-auc:{}".format(roc_auc_score(Y, y_pred_prob[:, 1])))
    return y_pred_prob


def plot_roc_curve(Y,Y_prob):
    fpr, tpr, thresholds = roc_curve(Y, Y_prob[:, 1])
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0,1],[0,1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Reciever Operating Characterstic')
    plt.legend()
    plt.show()
    accuracy_is = []
    # for i in thresholds:
    #     y_pred_auc = np.where(Y_pred_prob[:, 1] > i, 1, 0)
    #     accuracy_is.append(accuracy_score(Y_test, y_pred_auc, normalize=True))
    # accuracy_is = pd.concat([pd.Series(thresholds), pd.Series(accuracy_is)], axis=1)
    # accuracy_is.columns = ['Threshold', 'Accuracy']
    # accuracy_is.sort_values(by='Accuracy', ascending=False, inplace=True)
    # accuracy_is.head()


Dataset = Read_csv('/Users/navin.jain/Desktop/Learning 101/Basic Statistics/diabetes.csv')

Dataset = Data_types({'Age': 'int', 'Pregnancies': 'int', 'Glucose': 'float', 'BloodPressure': 'float',
                      'SkinThickness': 'float', 'Insulin': 'int', 'BMI': 'float',
                      'DiabetesPedigreeFunction': 'float', 'Outcome': 'bool'}, Dataset)

Dataset_copy = Dataset.copy()

Outlier = Outliers(Dataset_copy)

Dataset_copy = Replace_outliers(Dataset_copy)

Normal_dist_data = Gaussian_transform(Dataset_copy, 'Box Cox')

Normalised_Dataset = N_scaler(Normal_dist_data)

# X_train, X_test, Y_train, Y_test = X_Y_drop(Normalised_Dataset, 'Outcome', .4)
#
# Y_pred_prob = Log_regression(
#     X_train, Y_train, X_test, Y_test)
#
# plot_roc_curve(Y_test,Y_pred_prob)

y = Normalised_Dataset.iloc[:,-1]
x = Normalised_Dataset.iloc[:,:-1]

cv_model = LogisticRegression()
K_fold = KFold(7)
cv_result = cross_val_score(cv_model,x,y, cv=K_fold)
np.mean(cv_result)

skfold = StratifiedKFold(n_splits=5)
k_model = LogisticRegression()
sk_score = cross_val_score(k_model, x,y,cv=skfold)
np.mean(sk_score)