# introduction

- covert ex3data1.mat mat file to csv

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

- ex3weights mat file to csv

```python
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat('ex3weights.mat')

Theta1=pd.DataFrame(data['Theta1'])
Theta2=pd.DataFrame(data['Theta2'])

Theta1.to_csv('Theta1ex3weights.txt',header=False,index=False)
Theta2.to_csv('Theta2ex3weights.txt',header=False,index=False)
```

- ex4data1.mat file  to csv

```python
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat('ex4data1.mat')
X=pd.DataFrame(data['X'])
y=pd.DataFrame(data['y'])
X.to_csv('Xex4data1.txt',header=False,index=False)
y.to_csv('yex4data1.txt',header=False,index=False)
```
