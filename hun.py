import pandas as pd
import math 

from google.colab import drive
drive.mount('/content/drive')

df=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/dmw8-1.csv',delimiter='\t')
df=df.set_index('Day')

def entropy(df):
  pi=list(df['Play cricket'].value_counts())
  s=0
  su=sum(pi)
  for i in pi:
    val=i/su
    s+=val*math.log(val,2)
  return -s

## hunt s algorithm

data_sets=[]
data_sets.append((df.copy(),'main'))
count=0
while(len(data_sets)>0 and count<100):
  count+=1
  temp_df,parent_feature=data_sets.pop(0) 
  
  #print(temp_df,parent_feature,"hi",end="\n---------\n")
  if(len(temp_df)==0):
    continue
  best_column=(temp_df.columns)[0]
  print(f'\nparent feature:{parent_feature}\n')
  print('\nchoosing best column::',best_column,'\n')
  grouped_data=temp_df.groupby(best_column)
  for g,data in grouped_data:
    new_df=data.copy()
    new_df.drop([best_column],axis=1,inplace=True)
  #  print('new df\n',new_df,end='\n$$$$$$$$$$$$$$$$$$$$$\n')
    ## if all are in entropy zero print Just Yes or No
    print(f'node is {g}:',end="\n************\n")
    if(entropy(new_df)==0):
      print(new_df['Play cricket'].value_counts(),end="\n************\n")
     
    else:
      data_sets.append((new_df,parent_feature+'->' +g))
     # print(f'node is {g}: {ig_arr(new_df)}',end='\n*************\n')

