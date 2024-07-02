import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'  # prevent modules like `scipy` crash the KeyboradInterupt event