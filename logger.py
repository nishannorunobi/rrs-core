from email import message
import logging

#Creating and Configuring Logger

Log_Format = "%(levelname)s %(asctime)s - %(message)s"

logging.basicConfig(filename = "log/error_log.log",
                    filemode = "w",
                    format = Log_Format, 
                    level = logging.ERROR)

e_logger = logging.getLogger()

def error(errm):
    e_logger.error(errm)
#Testing our Logger

e_logger.error("Our First error Log Message")
