import logging

# Create a logger instance
logger = logging.getLogger(__name__)

# Set the logging level
logger.setLevel(logging.DEBUG)

# Create a file handler to write logs to a file
handler = logging.FileHandler('audit_trail.log')

# Set the logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(handler)
