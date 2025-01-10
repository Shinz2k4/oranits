import pandas as pd
import numpy as np
# Dãy số thập phân
values = [0.0339, 0.0342, 0.0397, 0.0320, 0.0351, 0.0347, 0.0313, 0.0277, 0.0309,
          0.0389, 0.0288, 0.0337, 0.0311, 0.0370, 0.0350, 0.0306, 0.0342, 0.0321,
          0.0293, 0.0322, 0.0328, 0.0329, 0.0301, 0.0397, 0.0290, 0.0329, 0.0365,
          0.0384, 0.0308, 0.0345]

# Tạo các khoảng phân loại
bins = np.linspace(min(values), max(values), num=4)

# Sử dụng pandas.cut để phân loại
categories = list(pd.cut(values, bins=bins, labels=[1, 2, 3], include_lowest=True))
