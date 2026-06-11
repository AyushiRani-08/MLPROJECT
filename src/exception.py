import sys
from src.logger import logging  # Tracking errors inside our log files

def error_message_detail(error, error_detail: sys):
    """
    Extracts explicit details about where the error occurred.
    """
    # exc_info() returns a tuple; the third element (execution traceback) holds the file and line numbers
    _, _, exc_tb = error_detail.exc_info()
    
    # Extracting the file name where the error happened
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Formatting a highly detailed error message
    error_message = (
        f"Error occurred in python script: [{file_name}] "
        f"at line number: [{exc_tb.tb_lineno}] "
        f"with error message: [{str(error)}]"
    )
    
    return error_message


class CustomException(Exception):
    """
    Custom exception class inheriting from the base Exception class.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        # Populate our custom detailed message using the function above
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        
    def __str__(self):
        # When printing the exception, return our detailed error message string
        return self.error_message