# auto_trace.py
import logging
import wrapt
import inspect

def setup_logging():
    logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@wrapt.decorator
def trace_calls(wrapped, instance, args, kwargs):
    logging.info(f"Entering: {wrapped.__name__}")
    result = wrapped(*args, **kwargs)
    logging.info(f"Exiting: {wrapped.__name__}")
    return result

def auto_instrument(module):
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            setattr(module, name, trace_calls(obj))
        elif inspect.isclass(obj):
            for cname, cobj in inspect.getmembers(obj):
                if inspect.isfunction(cobj) or inspect.ismethod(cobj):
                    setattr(obj, cname, trace_calls(cobj))
