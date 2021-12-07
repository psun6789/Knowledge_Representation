'''
Name            : Peter Sunny Shanthveer Markappa
Student No.     : R00208303
Assignment      : Knowledge Representation
Assignment No.  : 02
Submission Data : 05:December:2021
Final Submission

A2_COMP9016_Markappa_Peter_Sunny_R00208303

    Download Data set url:
    https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/

    Download White or Red Wine Quality file

    Procedures to follow after downloading the datasets

    I have used the white wine quality file..
    All the data are in Excel and they are in one column, so it has to be splitted
    Using filter option in excel you have to separate the data into different columns
    Remove the First row from the file which has name

    I have also attached the data file that can be used for demonstration

'''


# Importing

import os, sys, inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))

from probability import *
from sklearn.metrics import confusion_matrix
from learning import *
from probabilistic_learning import *
from notebook import *
import numpy as np
import csv
import pandas as pd

# BAYESIAN NETWORKS refering to the probability.ipybn file in aima-python
# Building Two different Models for given Random Variables
'''
-------- Model 1 -----------
• Traffic
• Fossil Fuels
• GlobalWarming
• Renewable Energy

-------- Model 2 -----------
• AI
• Employed

Here in my model I'm creating two different model as we can see that
Traffic, Fossil Fuels, Global Warming and Renewable are dependent on one another 
at same time AI and Employee are dependent.
The model diagram I have show in the Document Description that how each of them are related

'''
# -----------  Short Keys Used ---------------
# FF --> Fossil Fuels
# GW --> GlobalWarming
# RE --> Renewable Energy

# ------Short keys for True and False
# T = True
# F = False


# -----------------------------------------------------------------------------------------------------------
# 1.2 BAYESIAN NETWORKS START
# -----------------------------------------------------------------------------------------------------------
T, F = True, False


def bayesianNetworks():
    # Let us create Energy Model
    # Here Traffic and Fossil Fuel are independent Variable
    # Description and Truth Table of both the model is given in the document
    energy_model = BayesNet([
        ('Traffic', '', 0.001), ('FF', '', 0.002),
        ('GW', 'Traffic FF', {(T, T): 0.95, (T, F): 0.94, (F, T): 0.30, (F, F): 0.001}),
        ('RE', 'GW FF', {(T, T): 0.90, (T, F): 0.85, (F, T): 0.70, (F, F): 0.001})
    ])

    # Lets Create the Employee Model
    employee_model = BayesNet([
        ('AI', '', 0.5),
        ('Employed', 'AI',
         {T: 0.05, F: 0.90}
         )
    ])

    # ---------------------------------------------------
    # Print the Nodes of both the models
    # ---------------------------------------------------
    print("  ")
    print("-------------------# Associated Conditional Probability Tables #----------------------------------")
    print("*******************************************************************************************")
    print(energy_model.variable_node('Traffic').cpt)
    print(energy_model.variable_node('RE').cpt)
    print(energy_model.variable_node('GW').cpt)
    print(employee_model.variable_node('AI').cpt)
    print(employee_model.variable_node('Employed').cpt)
    print("  ")
    # ---------------------------------------------------
    # Demonstration of querying the network.
    # ---------------------------------------------------
    print("*******************************************************************************************")
    print("------------------------# Querying of Energy Model #---------------------------------------")
    print("*******************************************************************************************")
    # Querying of Energy Model
    print(energy_model.nodes[2].p(T, {'Traffic': T, 'FF': T}))
    print(energy_model.nodes[2].p(T, {'Traffic': T, 'FF': F}))
    print(energy_model.nodes[2].p(T, {'Traffic': F, 'FF': T}))
    print(energy_model.nodes[2].p(T, {'Traffic': F, 'FF': F}))
    print(energy_model.nodes[3].p(F, {'GW': T, 'FF': T}))
    print(energy_model.nodes[3].p(F, {'GW': T, 'FF': F}))
    print(energy_model.nodes[3].p(F, {'GW': F, 'FF': T}))
    print(energy_model.nodes[3].p(F, {'GW': F, 'FF': F}))

    print("*******************************************************************************************")
    print("------------------------# Querying of Employee Model #---------------------------------------")
    print("*******************************************************************************************")
    # Querying of Employee Model
    print(employee_model.nodes[1].p(T, {'AI': T}))
    print(employee_model.nodes[1].p(T, {'AI': F}))
    print(employee_model.nodes[1].p(F, {'AI': T}))
    print(employee_model.nodes[1].p(F, {'AI': F}))

# -----------------------------------------------------------------------------------------------------------
# 1.2 BAYESIAN NETWORKS END
# -----------------------------------------------------------------------------------------------------------




# -----------------------------------------------------------------------------------------------------------
# 1.3.1 NAIVE BAYES START
# -----------------------------------------------------------------------------------------------------------
def developingEvaluationModel(cmcData):
    wineQualityDataArray = np.array(cmcData.examples)
    probability_prior_table = dict()
    probabilityofLikelyhood, frequency_dictionary, probabilityOfEvidence = [], [], []
    windQualityShape = (wineQualityDataArray.shape[1] - 1)


    # Calculation of the Prior Probability of Each Quality(Class) by using formula of Prior Probability
    # Prior Probability = length of each quality(Class) / total length
    # storing into the probability table which is dictionary
    probability_prior_table[0] = len(wineQualityDataArray[wineQualityDataArray[:, -1] == 0]) / wineQualityDataArray.shape[0]
    probability_prior_table[1] = len(wineQualityDataArray[wineQualityDataArray[:, -1] == 1]) / wineQualityDataArray.shape[0]
    probability_prior_table[2] = len(wineQualityDataArray[wineQualityDataArray[:, -1] == 2]) / wineQualityDataArray.shape[0]
    probability_prior_table[3] = len(wineQualityDataArray[wineQualityDataArray[:, -1] == 3]) / wineQualityDataArray.shape[0]
    probability_prior_table[4] = len(wineQualityDataArray[wineQualityDataArray[:, -1] == 4]) / wineQualityDataArray.shape[0]
    probability_prior_table[5] = len(wineQualityDataArray[wineQualityDataArray[:, -1] == 5]) / wineQualityDataArray.shape[0]
    probability_prior_table[6] = len(wineQualityDataArray[wineQualityDataArray[:, -1] == 6]) / wineQualityDataArray.shape[0]
    probability_prior_table[7] = len(wineQualityDataArray[wineQualityDataArray[:, -1] == 7]) / wineQualityDataArray.shape[0]
    probability_prior_table[8] = len(wineQualityDataArray[wineQualityDataArray[:, -1] == 8]) / wineQualityDataArray.shape[0]
    probability_prior_table[9] = len(wineQualityDataArray[wineQualityDataArray[:, -1] == 9]) / wineQualityDataArray.shape[0]
    probability_prior_table[10] = len(wineQualityDataArray[wineQualityDataArray[:, -1] == 10]) / wineQualityDataArray.shape[0]



    #  ---------------------------Compute the Probability of evidences----------------------------------

    for val in range(windQualityShape):
        uniquevalue, frequency = np.unique(wineQualityDataArray[:, val], return_counts=True)
        frequency_dictionary.append({uniquevalue[j]: frequency[j] for j in range(len(uniquevalue))})

    for i in range(wineQualityDataArray.shape[0]):
        prod = 1
        for j in range(wineQualityDataArray.shape[1] - 1): prod *= (frequency_dictionary[j][
            wineQualityDataArray[i, j]]) / (wineQualityDataArray.shape[0])
        probabilityOfEvidence.append(prod)


    #  ---------------------------Compute the probability of likelihood of evidences----------------------------------

    targetClass = cmcData.values[cmcData.target]
    # CountingProbDist is defined in aima-python in learning.ipybn.. I have refered from this line and implemented
    # attr_dists = {(gv, attr): CountingProbDist(dataset.values[attr]) for gv in target_vals for attr in dataset.inputs}
    attributes = {(key, values): CountingProbDist(cmcData.values[values]) for key in targetClass for values in
                  cmcData.inputs}
    for example in cmcData.examples: probabilityofLikelyhood.append(
        probability_prior_table[example[-1]] * product(attributes[example[-1], i][example[i]] for i in cmcData.inputs))



    #  --------------------------- Printing all the values----------------------------------
    print("probability_prior_table = ", "\n", probability_prior_table, "\n", "\n", "\n", "\n")
    print("probability of Likelyhood of Evidence = ","\n", probabilityofLikelyhood, "\n", "\n", "\n", "\n")
    print("Probability of Evidence = ","\n", probabilityOfEvidence, "\n", "\n", "\n", "\n")

# -----------------------------------------------------------------------------------------------------------
# 1.3.1 NAIVE BAYES END
# -----------------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------------------------
# 1.3.2 NAIVE BAYES LEARNER START
# -----------------------------------------------------------------------------------------------------------

def naiveBaseLearner(cmcData):
    classifier = NaiveBayesDiscrete(cmcData)
    dataarray = np.array(cmcData.examples)[:, -1]

    predict = [classifier(data) for data in cmcData.examples]

    print(f'Accuracy  = {(np.sum(dataarray == predict) / len(predict)) * 100}')
    # I'm using sklearn library confusion matrix to find the confustion matrix of the iris dataset
    print("Confusion Matrix = ", "\n", confusion_matrix(dataarray, predict))

# ---------------------------------------------------
# 1.3.2 NAIVE BAYES LEARNER ENDS
# ---------------------------------------------------




# ---------------------------------------------------
# Main Function
# ---------------------------------------------------

if __name__ == "__main__":

    print("____________________________Executing the 1.2 BAYESIAN NETWORKS START ____________________________")
    bayesianNetworks()
    print("____________________________Executing the 1.2 BAYESIAN NETWORKS END ____________________________")

    print("-----------------------------------------------------------------------------------------------")


    print("__________________________Executing the 1.3.1 NAIVE BAYES START __________________________")
    # I have used White Wine Quality CSV File
    # https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
    # After Downloading the Wite wine -- "winequality-white" this will be the file name, keep as it is

    ''' I'm Passing the File name here to read it, because all the data is in single row, so using pandas I'm splitting
        it and save it back to csv '''

    account = pd.read_csv("aima-data/winequality-white.csv", delimiter=';')

    account.to_csv('aima-data/WhiteWine.csv', index=None, header=False)

    wineQuality = DataSet(name='WhiteWine')

    developingEvaluationModel(wineQuality)

    print("__________________________Executing the 1.3.1 NAIVE BAYES END __________________________")
    print("-----------------------------------------------------------------------------------------------")





    print("__________________________Executing the 1.3.2 NAIVE BAYES LEARNER START __________________________")
    # Part 1 of 1.3.2  Performance
    naiveBaseLearner(wineQuality)
    print("__________________________Executing the 1.3.2 NAIVE BAYES LEARNER END __________________________")

    # https://archive-beta.ics.uci.edu/ml/datasets/wine+quality
    # https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/