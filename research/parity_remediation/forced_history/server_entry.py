import runpy
from observer import install
install()
runpy.run_module('sglang.launch_server',run_name='__main__')
