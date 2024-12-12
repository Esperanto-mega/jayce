# In[Import]
from functools import wraps

# In[Overload]
def overload(func):
    @wraps(func)
    def wrapper(*args,**kargs):
        if len(args) == 2:
            # For input like model(g).
            g = args[1]
            return func(args[0],
                        g.x,
                        g.edge_index,
                        g.edge_attr,
                        g.batch)
        elif len(args) == 5:
            # For input like model(x,edge_index,edge_attr,batch).
            return func(*args)
        elif len(args) == 3:
            # model(g, edge_weight)
            g = args[1]
            edge_weight = args[2]
            return func(args[0],g.x,g.edge_index,edge_weight,g.batch)
        else:
            raise TypeError
    return wrapper