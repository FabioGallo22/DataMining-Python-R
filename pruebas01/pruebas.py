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
print("Hello")

bank_train = pd.read_csv("../datasets/bank_marketing_training") # pag 12
print("Bank_train: \n", bank_train)

print("_______\n Dos columnas: previous_outcome  y  response")
crosstab_01 = pd.crosstab(bank_train['previous_outcome'], bank_train['response']) # pag 13
print(crosstab_01)


print("_________________****")
arr = [22,33,44,55,66,77,88,99,00]
print("Some values 1: ", arr[2:6])
# print("Some values 2: ", arr.loc[[2,6,7]]) loc es de pandas y solo sirve si creo la estructura con pd.crosstab()