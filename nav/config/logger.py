from pdslogger import PdsLogger


# Default logger instance for image processing operations
DEFAULT_LOGGER = PdsLogger('default', lognames=False, digits=3)

# TODO Clean this up - don't do anything on import
# DEFAULT_LOGGER.add_handler(STDOUT_HANDLER)
# DEFAULT_LOGGER.info('*** START OF LOG ***')
