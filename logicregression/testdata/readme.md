# introduction

- covert mat file to csv

    ```python
    import numpy as np
    import pandas as pd
    from scipy.io import loadmat
    import matplotlib.pyplot as plt

    data = loadmat('ex3data1.mat')

    X=pd.DataFrame(data['X'])
    y=pd.DataFrame(data['y'])

    X.to_csv('Xex3data1.txt',header=False,index=False)
    y.to_csv('yex3data1.txt',header=False,index=False)
    ```
