from pdslogger import PdsLogger, STDOUT_HANDLER


# Default logger instance for image processing operations
DEFAULT_LOGGER = PdsLogger('default', lognames=False)
DEFAULT_LOGGER.add_handler(STDOUT_HANDLER)
DEFAULT_LOGGER.info('*** START OF LOG ***')
