import sys

def error_message_detail(error_message, error_detail: sys):
    _, _, exe_tb = error_detail.exc_info()

    if exe_tb is not None:
        file_name = exe_tb.tb_frame.f_code.co_filename
        error_message = (
            f"Error occurred in python script name [{file_name}] "
            f"line number [{exe_tb.tb_lineno}] "
            f"error message [{str(error_message)}]"
        )
    else:
        # Handles manually raised errors (no traceback)
        error_message = f"Error message: {str(error_message)}"

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
