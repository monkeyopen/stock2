import inspect
import logging
import os
from dotenv import load_dotenv

load_dotenv()
LOG_LEVEL = os.getenv('LOG_LEVEL')
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=LOG_LEVEL)


def log(str=""):
    # Get the frame of the caller
    frame = inspect.currentframe().f_back
    # Get the name of the function and the line number
    function_name = frame.f_code.co_name
    line_number = frame.f_lineno
    filename = frame.f_code.co_filename
    # Return the function name, line number, and filename
    pre_str = f"{filename},line {line_number}, {function_name}() {str}"
    # print(pre_str)

    logging.debug(pre_str)
    # logging.info(pre_str + 'info 信息')
    # logging.warning(pre_str + 'warning 信息')
    # logging.error(pre_str + 'error 信息')
    # logging.critical(pre_str + 'critial 信息')
