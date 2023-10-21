import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import DataPrep.normalizeDataset as Data

def inverseNormalize(predictions, NValsPath):
    NValuesList = pd.read_csv(NValsPath)
    #print(NValuesList)

    for i, rows in enumerate(predictions):
        predictions[i,:] = predictions[i,:]*NValuesList.loc[i,'Values']

    return predictions

#print(inverseNormalize(Data.output_tensors, 'NValsFile28K.csv'))