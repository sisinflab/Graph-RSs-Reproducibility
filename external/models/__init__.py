def import_model_by_backend(tensorflow_cmd, pytorch_cmd):
    import sys
    for _backend in sys.modules["external"].backend:
        if _backend == "tensorflow":
            exec(tensorflow_cmd)
        elif _backend == "pytorch":
            exec(pytorch_cmd)
            break

import sys
for _backend in sys.modules["external"].backend:
    if _backend == "tensorflow":
        pass
    elif _backend == "pytorch":
        from .ngcf.NGCF import NGCF
        from .lightgcn.LightGCN import LightGCN
        from .dgcf.DGCF import DGCF
        from .sgl.SGL import SGL
        from .ultragcn import UltraGCN
        from .gfcf import GFCF
        from .sgl import SGL
