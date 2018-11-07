

```python
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat('ex3weights.mat')

Theta1=pd.DataFrame(data['Theta1'])
Theta2=pd.DataFrame(data['Theta2'])

Theta1.to_csv('Theta1ex3weights.txt',header=False,index=False)
Theta1.to_csv('Theta1ex3weights',header=False,index=False)
```