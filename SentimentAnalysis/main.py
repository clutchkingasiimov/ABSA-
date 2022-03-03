import pandas as pd 



data = pd.read_csv('/home/sauraj/Desktop/LCSTextDetection/exercise2/data/devdata.csv',delimiter='\t',
header=None)

#Rename columns 
data = data.rename(columns={
    0:'polarity',
    1:'aspect_cat',
    2:'target_term',
    3:'char_offset',
    4:'sentence'
})

print(data.head())