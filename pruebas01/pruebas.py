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
plt.show()

"""
    Capítulo 3
"""
bank_train = pd.read_csv("../datasets/bank_marketing_training") # 31
print("\n Shape: ", bank_train.shape) # Pag 32. To find the number of rows and columns in the dataset.

bank_train['index'] = pd.Series(range(0,26874)) # pag 32

print("Head: ", bank_train.head) # pag 32

# 3.5.1 how to Change Misleading Field Values Using python, Pag 34
print("\n Before replacement of  999 for NaN:\n", bank_train['days_since_previous'])
bank_train['days_since_previous'] = bank_train['days_since_previous'].replace({999: np.NaN})
print("\n After replacement of 999 for NaN:\n", bank_train['days_since_previous'])
print("_______\n ")