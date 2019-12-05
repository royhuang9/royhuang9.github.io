# How to debug in jupyter notebook

http://www.blog.pythonlibrary.org/2018/10/17/jupyter-notebook-debugging/

## Using pdb
You can put "pdb.set_trace()" or "breakpoint()" (supported by Python 3.7) in the code.
For example

``` python
def bad_function(var):
    import pdb
    pdb.set_trace()
    return var + 0
 
bad_function("Mike")
```
and
```
def bad_function(var):
    breakpoint()
    return var + 0
 
bad_function("Mike")
```

### Commands
You can use any of pdb’s command right inside of your Jupyter Notebook. Here are some examples:

w(here) – Print the stack trace
d(own) – Move the current frame X number of levels down. Defaults to one.
u(p) – Move the current frame X number of levels up. Defaults to one.
b(reak) – With a *lineno* argument, set a break point at that line number in the current file / context
s(tep) – Execute the current line and stop at the next possible line
c(ontinue) – Continue execution

## ipdb
there is an IPython debugger that we can use called IPython.core.debugger.set_trace. Let’s create a cell with the following code:

```python
from IPython.core.debugger import set_trace
 
def bad_function(var):
    set_trace()
    return var + 0
 
bad_function("Mike")
```

Or when an exception happen, open a new window, type "%pdb" or "%debug"
You can also type them in the beginning of a cell.


## What about %%debug?
There is yet another way that you can open up a debugger in your Notebook. You can use `%%debug` to debug the entire cell like this:

%%debug
 
def bad_function(var):
    return var + 0
 
bad_function("Mike")
This will start the debugging session immediately when you run the cell. What that means is that you would want to use some of the commands that pdb supports to step into the code and examine the function or variables as needed.

Note that you could also use `%debug` if you want to debug a single line.

