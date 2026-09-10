

def RK4(f,y,h,t):
    k1 = f(y,t)
    k2 = f(y + 0.5*h*k1, t + 0.5*h)
    k3 = f(y + 0.5*h*k2, t + 0.5*h)
    k4 = f(y + h*k3, t + h)
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

