import sys
import signal
from contextlib import contextmanager

class Color: #pylint: disable=W0232
    GRAY=30
    RED=31
    GREEN=32
    YELLOW=33
    BLUE=34
    MAGENTA=35
    CYAN=36
    WHITE=37
    CRIMSON=38    

def colorize(num, string, bold=False, highlight = False):
    assert isinstance(num, int)
    attr = []
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def colorprint(colorcode, text, o=sys.stdout, bold=False):
    o.write(colorize(colorcode, text, bold=bold))

def warn(msg):
    print (colorize(Color.YELLOW, msg))

def error(msg):
    print (colorize(Color.RED, msg))

# http://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call-in-python
class TimeoutException(Exception): pass
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException(colorize(Color.RED, "   *** Timed out!", highlight=True))
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
