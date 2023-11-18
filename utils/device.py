import sympy as sp
from sympy import IndexedBase

sqrt2 = sp.sqrt(2)

def DC0011(p1:IndexedBase,p2:IndexedBase):
    '''
    Preprare the initial state |00>+|11> on p1 and p2
    '''

    expr = (p1[0]*p2[0]+p1[1]*p2[1])/sqrt2

    return expr.expand()

def DC00(p1:IndexedBase,p2:IndexedBase):
    '''
    Preprare the initial state |00> on p1 and p2
    '''

    expr = p1[0]*p2[0]

    return expr.expand()

def DC11(p1:IndexedBase,p2:IndexedBase):
    '''
    Preprare the initial state |11> on p1 and p2
    '''

    expr = p1[1]*p2[1]

    return expr.expand()

def BS(expr,p1:IndexedBase,p2:IndexedBase):
    '''
    Apply beam splitter on path p1 and path p2
    '''
    subrule = [(p1[x], (p1[x]+sp.I*p2[x])/sp.sqrt(2) ) for x in [0,1]] + [(p2[x], (p2[x]+sp.I*p1[x])/sp.sqrt(2) ) for x in [0,1]]
    expr = expr.subs(subrule, simultaneous=True).expand()
    #print(expr)
    return expr

def PBS(expr, p1:IndexedBase,p2:IndexedBase):
    '''
    Apply polarized beam splitter on path p1 and path p2
    '''
    subrule = [(p1[0],p1[0]),(p1[1], p2[1]),(p2[0],p2[0]), (p2[1],p1[1])]
    expr = expr.subs(subrule, simultaneous=True).expand()
    #print(expr)
    return expr

def HWP(expr, p1:IndexedBase, theta:float):
    '''
    Apply Half-Wave Plate on path p
    '''
    subrule = [(p1[0], (sp.cos(2*theta)*p1[0] + sp.sin(2*theta)*p1[1] )), (p1[1], (sp.sin(2*theta)*p1[0] - sp.cos(2*theta)*p1[1] ))]
    expr = expr.subs(subrule, simultaneous=True).expand()
    #print(expr)
    return expr

def QWP(expr, p1:IndexedBase, theta:float):
    '''
    Apply Quarter-Wave Plate on path p
    '''
        

    subrule = [(p1[0], ( (1-sp.I*sp.cos(2*theta))*p1[0] - sp.I*sp.sin(2*theta)*p1[1] )/sqrt2), (p1[1], ( -sp.I*sp.sin(2*theta)*p1[0] + (1+sp.I*sp.cos(2*theta))*p1[1] )/sqrt2 )]
    expr = expr.subs(subrule, simultaneous=True).expand()
    #print(expr)
    return expr

def R(expr, p1):
    '''
    Reflection on path p
    '''
    #subrule = [(p[0],sp.I*p[0]), (p[1],sp.I*p[1])]
    ii = sp.Wild('ii')
    expr = expr.replace(p1[ii], sp.I*p1[ii], simultaneous=True).expand()
    return expr

