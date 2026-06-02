"""融合方法自动发现。

在此 __init__ 中导入各方法模块以触发注册。
"""

from . import baseline_cnn
from . import template
from . import cnmf

# 后续方法在实现后取消注释：
# from . import fusionmamba
# from . import hypertransformer
# from . import dstf
# from . import umsft
# from . import csakd
# from . import hsr-kan
