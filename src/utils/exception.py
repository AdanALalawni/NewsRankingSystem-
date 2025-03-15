import sys
import os
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),"../"))
sys.path.append(MAIN_DIR)

def error_masseges(error, error_detail:sys):
    """Create Custom error masseges"""
    _,_,exp_tb = error_detail.exc_info()
    error_massege= f" Error: in file {exp_tb.tb_frame.f_code.co_filename} in line {exp_tb.tb_lineno} error massege {error}"
    return error_massege
class CustomException(Exception):
    def __init__(self, error_massege,error_detail:sys):
        super().__init__(error_massege)
        self.error_massege= error_masseges(error_massege, error_detail= error_detail)
    def __str__(self):
        return self.error_massege