import pandas as pd 
from text_processor import TextProcessor


data = pd.read_csv('/home/sauraj/ABSA-/exercise2/data/devdata.csv',delimiter='\t',
header=None)

#Rename columns 
data = data.rename(columns={
    0:'polarity',
    1:'aspect_cat',
    2:'target_term',
    3:'char_offset',
    4:'sentence'
})

tp = TextProcessor(data['sentence']) #Sentence column 
data['processed_sentence'] = tp.Preprocess_Text()

# print(data.head())