# Libro: Data science using Python and R (2019).pdf
# Donde descargar datasets: http://www.dataminingconsultant.com/
#    Donde ver la finculación inicial del proyecto a GIT https://www.geeksforgeeks.org/how-to-upload-project-on-github-from-pycharm/
#              VCS>Enable Version Control Integration.

# From https://www.jetbrains.com/help/pycharm/commit-and-push-changes.html
#   Commit Ctrl+K
#   Commit and Push Ctrl+Alt+K
#   Push Ctrl+Shift+K

"""
    Capítulo 2
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split
import random

show_graphics = False
print("Hello")

bank_train = pd.read_csv("../datasets/bank_marketing_training") # pag 12
print("Bank_train: \n", bank_train)

print("_______\n Dos columnas: previous_outcome  y  response")
crosstab_01 = pd.crosstab(bank_train['previous_outcome'], bank_train['response']) # pag 14
print(crosstab_01)

print("_______\n bank_train.loc[0]")
print(bank_train.loc[0])

print("_______\n bank_train.loc[[0, 2, 3]]")
print(bank_train.loc[[0, 2, 3]])

print("_______\n bank_train[0:10]")
print(bank_train[0:10])

print("_______\n bank_train['age']")
print(bank_train['age'])

print("_______\n bank_train[['age','job']]")
print(bank_train[['age','job']]) # pag 15

print("_______\n Gráficos") # pag 17, es necesario añadir la libreria matplotlib
bank_train['age'].plot(kind='hist')
if show_graphics:
    plt.show()

"""
    Capítulo 3
"""
bank_train = pd.read_csv("../datasets/bank_marketing_training") # 31
print("\n Shape: ", bank_train.shape) # Pag 32. To find the number of rows and columns in the dataset.

bank_train['index'] = pd.Series(range(0,26874)) # pag 32

print("Head: ", bank_train.head) # pag 32

# 3.5.1 how to Change Misleading Field Values Using python, Pag 34
#print("\n Before replacement of  999 for NaN:\n", bank_train['days_since_previous'])
bank_train['days_since_previous'].plot(kind = 'hist',title = 'Histogram of Days Since Previous (before replacement 999 for NaN)')
if show_graphics:
    plt.show()

bank_train['days_since_previous'] = bank_train['days_since_previous'].replace({999: np.NaN})

#print("\n After replacement of 999 for NaN:\n", bank_train['days_since_previous'])
bank_train['days_since_previous'].plot(kind = 'hist',title = 'Histogram of Days Since Previous (before replacement 999 for NaN)')
if show_graphics:
    plt.show()

# 3.6.1 how to reexpress Categorical Field Values Using python. Pag 36
bank_train['education_numeric'] = bank_train['education']
print("Head after adding column 'education_numeric': ", bank_train.head)
print("\n 'education_numeric' BEFORE using dictionary: ", bank_train['education_numeric'])
dict_edu = {"education_numeric":
                {"illiterate": 0,
                 "basic.4y": 4,
                 "basic.6y": 6,
                 "basic.9y": 9,
                 "high.school": 12,
                 "professional.course": 12,
                 "university.degree": 16,
                 "unknown": np.NaN}}
bank_train.replace(dict_edu, inplace=True) # pag 38 >>>>>>>
print("\n 'education_numeric' AFTER using dictionary: ", bank_train['education_numeric'])

# 3.7.1 how to Standardize Numeric Fields Using python. Pag 40
bank_train['age_z'] = stats.zscore(bank_train['age'])
print("\n standardized 'age' as a new variable, 'age_z': ", bank_train['age_z'])

# 3.8 IDeNtIFYING OUtLIerS. Pag 40
# Es 3 porq como se creó una distribución de media 0 y desvío 1, y dice q +-3 veces el desvío, los que están afuera son outliers
bank_train_outliers = bank_train.query('age_z > 3 | age_z < -3')
print("\n OUTLIERS: ", bank_train_outliers['age_z'])
bank_train_sort = bank_train.sort_values(['age_z'], ascending=False)
print("\n Age_z sorted: ", bank_train_sort)
print("\n report the age and marital status of the 15 people who have the largest age_z values:",
      bank_train_sort[['age', 'marital']].head(n=15)) # pag 42

"""
    Capítulo 4
"""
# 4.2.1 how to Construct a Bar Graph with Overlay Using python. Pag 49
# First, create a contingency table of the values in the predictor and target variables
crosstab_01 = pd.crosstab(bank_train['previous_outcome'], bank_train['response'])
crosstab_01.plot(kind='bar', stacked = True)
if show_graphics:
    plt.show()
crosstab_norm = crosstab_01.div(crosstab_01.sum(1), axis=0) # To normalize. Pag 49
crosstab_norm.plot(kind='bar', stacked = True)
if show_graphics:
    plt.show()

# 4.3.1 how to Construct Contingency tables Using python. Pag 52
crosstab_02 = pd.crosstab(bank_train['response'], bank_train['previous_outcome'])
print("\n Contengency table: \n", crosstab_02)
print("\n Contengency table (round): \n", round(crosstab_02.div(crosstab_02.sum(0), axis = 1)*100, 1)) # pag 52

#__________________________________________________________________________
# 4.4.1 how to Construct histograms with Overlay Using python. Pag 55
bt_age_y = bank_train[bank_train.response == "yes"]['age']
bt_age_n = bank_train[bank_train.response == "no"]['age']

plt.close('all')
plt.hist([bt_age_y, bt_age_n], bins = 10, stacked = True)
plt.legend(['Response = Yes', 'Response = No'])
plt.title('Histogram of Age with Response Overlay')
plt.xlabel('Age')
plt.ylabel('Frequency')
if show_graphics:
    plt.show()

# create a normalized histogram
(n, bins, patches) = plt.hist([bt_age_y, bt_age_n], bins = 10, stacked = True)
# n is the height of the histogram bars and bins are the
# boundaries of each bin in the histogram. Note that, since two variables are being
# plotted in the histogram, n has two series of numbers. The first series is for the first
# variable and the second one is for the second variable. The first number in each
# series is the height of the first bar for each variable.

# n_table, is a two‐column matrix where each column’s entries hold the heights of each bar
n_table = np.column_stack((n[0], n[1]))
# To calculate what proportion of the bar is accounted for by each variable, we need to divide each row by the sum across that row.
n_norm = n_table / n_table.sum(axis=1)[:, None]
# we create an array whose rows are the exact cuts of each bin
ourbins = np.column_stack((bins[0:10], bins[1:11])) # Each row in ourbins gives the upper and lower bounds of each bin

# Now, we are ready to create our normalized histogram.
plt.close('all')
p1 = plt.bar(x = ourbins[:,0], height = n_norm[:,0], width = ourbins[:, 1] - ourbins[:, 0])
p2 = plt.bar(x = ourbins[:,0], height = n_norm[:,1], width = ourbins[:, 1] - ourbins[:, 0], bottom = n_norm[:,0])
plt.legend(['Response = Yes', 'Response = No'])
plt.title('Normalized Histogram of Age with Response Overlay')
plt.xlabel('Age')
plt.ylabel('Proportion')
if show_graphics:
    plt.show()

#__________________________________________________________________________
# 4.5.1 how to perform Binning Based on predictive Value Using python. Pag 59
#        Bin the values using cut() from the pandas package.
plt.close('all')
bank_train['age_binned'] = pd.cut(x = bank_train['age'], bins = [0, 27, 60.01, 100], labels=["Under 27", "27 to 60", "Over 60"], right =False)
print("\nNew column 'age_binned' and 'age':\n", bank_train[['age_binned', 'age']])

crosstab_02 = pd.crosstab(bank_train['age_binned'], bank_train['response'])
crosstab_02.plot(kind='bar', stacked = True, title = 'Bar Graph of Age (Binned) with Response Overlay')
if show_graphics:
    plt.show()

"""
    Capítulo 5
    PREPARING TO MODEL THE DATA
"""
# 5.2.1 how to partition the Data in python. Pag 70
bank = pd.read_csv("../datasets/bank-additional.csv", delimiter=';')
print("bank: \n", bank)

bank_train, bank_test = train_test_split(bank, test_size = 0.25, random_state = 7)

print("bank.shape (original):\n", bank.shape)
print("bank_train.shape: \n", bank_train.shape)
print("bank_test.shape: \n", bank_test.shape)

# donde quedé 5.3 VaLIDatING YOUr partItION
#       Deriva a leer otro libro "Data Mining and Predictive Analytics", en la pag 148 de ese libro dice exactamente lo mismo
#           Depending on the variable types involved,
#           different statistical tests are required.
#           For a numerical variable, use the two‐sample t‐test for the difference in means.
#           For a categorical variable with two classes, use the two‐sample Z‐test for the
#           difference in proportions.
#           For a categorical variable with more than two classes, use the test for the
#           homogeneity of proportions.


# 5.4.1 how to Balance the training Data Set in python. Pag 74
# First, count how many "yes" are in response
print("bank_train['response'].value_counts(): \n", bank_train['response'].value_counts())
# The loc() command subsets the bank_train data based on the condition bank_train[‘response’] == “yes”
to_resample = bank_train.loc[bank_train['response'] == "yes"]
# sample from our records of interest
our_resample = to_resample.sample(n = 841, replace = True) # 841 is the result of a formula according to the porcentaje of "yes" wanted.

print("_______\n ")

