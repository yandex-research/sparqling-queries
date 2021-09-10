import traceback
import signal
import re
import os
import errno
import random
import numpy as np

from contextlib import contextmanager


class TimeoutException(Exception): pass

class SparqlGenerationError(Exception): pass

class SparqlRunError(Exception): pass

class SqlRunError(Exception): pass

class SparqlWrongAnswerError(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def handle_exception_sparql_process(e, verbose=False):
    sparql_exception_types = [SparqlGenerationError, SparqlRunError, SqlRunError, SparqlWrongAnswerError]
    if type(e) in sparql_exception_types:
        # have one one the special type that should be caused by something
        sparql_error_type = type(e).__name__
        if verbose:
            print("SPARQL error type", sparql_error_type)
        cause_analisys = handle_exception(e.__cause__, verbose=verbose)
        cause_analisys["sparql_error_type"] = sparql_error_type
        return cause_analisys
    else:
        return handle_exception(e, verbose=verbose)


def handle_exception(e, verbose=False):
    '''Get all information about exception.
    '''
    # exc_type, exc_val, exc_tb = sys.exc_info()
    # instead all using the top exception being handled analyze the passed e object
    exc_type, exc_val, exc_tb =  type(e), e, e.__traceback__

    format_exc = traceback.format_exception(exc_type, exc_val, exc_tb)
    full_message = ' '.join(format_exc)
    file_name = ''
    module_name = ''
    line_num = -1
    # find file name and line number
    for exc in format_exc:
        if exc.find('File') >= 0 and exc[exc.find('File') + 5] == '\"': # 5 = len('File ')
            exc = exc.split('\n')[0] # first line contains all info
            if exc.find('/python') >= 0:
                continue
            cur_file_name = exc.split('\"')[1].split('/')[-1] # "abs_path/file_name"
            file_name = cur_file_name
            module_name = exc[exc.find(', in '):].split(' ')[-1] # "in module_name"
            line_num = re.findall('line \d+', exc)[0].split(' ')[-1] # "line num_line"

    assert int(line_num) >= 0, 'unknown line number'
    assert file_name, 'unknown file name'

    if verbose:
        print('Error info:')
        print('message:', str(e))
        print('type:', exc_type.__name__)
        print('file:', file_name)
        print('module_name:', module_name)
        print('line_number:', line_num)
        #print(full_message)

    return {'message': str(e), 'type': exc_type.__name__, 'file': file_name,
            'module_name': module_name, 'line_number': line_num}


def mkdir(path):
    """From https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/miscellaneous.py
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def set_random_seed(random_seed, cuda=False):
    random.seed(random_seed)
    np.random.seed(random_seed)
    try:
        import torch
        torch.manual_seed(random_seed)
        if cuda:
            torch.cuda.manual_seed_all(random_seed)
    except:
        pass
