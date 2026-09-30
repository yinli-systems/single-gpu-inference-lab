import runpy
from observer import install
install()
if __name__=='__main__':
 runpy.run_module('sglang.launch_server',run_name='__main__')
