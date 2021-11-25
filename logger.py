#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

logging.basicConfig(format='%(created)-5.2f %(levelname)-5.5s [%(funcName)8s] %(message)s',
                    level=logging.INFO)

logger_1 = logging.getLogger('SynthText Logger')

log_file = "logfile.log"
log_level = logging.DEBUG
logging.basicConfig(level=log_level, filename=log_file, filemode="w+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
logger = logging.getLogger("baker_logger")


def wrap(pre, post):
	""" Wrapper """
	def decorate(func):
		""" Decorator """
		def call(*args, **kwargs):
			""" Actual wrapping """
			pre(func)
			result = func(*args, **kwargs)
			post(func)
			return result
		return call
	return decorate

def entering(func):
   # print("Entered {}".format(func.__name__))
    logger.debug("Entered %s", func.__name__)

def exiting(func):
    #print("Exited  {}".format(func.__name__))
    logger.debug("Exited  %s", func.__name__)