import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    Creates a detailed error message.

    Args:
        error (Exception): The exception object.
        error_detail (sys): The sys module to get exception info.

    Returns:
        str: A formatted error message with script name, line number, and error.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in python script name [{file_name}] line number [{line_number}] error message [{str(error)}]"
    return error_message

class CustomException(Exception):
    """
    Custom exception class that logs the detailed error message.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        # Log the detailed error message
        logging.error(self.error_message)

    def __str__(self):
        return self.error_message

if __name__ == "__main__":
    # Example of how to use the custom exception
    try:
        a = 1 / 0
    except Exception as e:
        raise CustomException(e, sys)
