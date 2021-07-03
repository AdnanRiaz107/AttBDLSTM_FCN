import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
import os


#%matplotlib inline
sns.set_context('notebook')
#%config InlineBackend.figure_format = 'retina'
root="H:\\revised updated thesis\\TL18 Training\\"

f_list=os.listdir(root)
print(f_list)
plt.figure(figsize=(8,6))
plt.ylabel('Training Loss')
plt.xlabel('Epoch')
plt.grid(True)
count=1

#f[:4]
for f in f_list:
    df= pd.read_json(root+f)
    plt.plot(df[1],df[2],label= (f[:-5]))
    count+=1
plt.legend(loc='best')
p=plt.show()

