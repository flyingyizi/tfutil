# introduction

- ex7data2.mat file  to csv

```python
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat('ex7data2.mat')
X=pd.DataFrame(data['X'])
X.to_csv('Xex7data2.txt',header=False,index=False)
```
# 工具集

|tool| descripton|
|--|--|
|monitor|采集机器信息作为机器异常检测数据源|
|||