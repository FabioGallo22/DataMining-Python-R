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

print("_______\n ")
