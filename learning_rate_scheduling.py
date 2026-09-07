import math
import torch 


def learning_rate_schedule(t, a_max, a_min, T_w, T_c): 
    ''' 
        @parameter: 
            t: current iteration 
            a_max: Maximum learning rate 
            a_min: Miniumum learning rate
            T_w: num of warm-up iterations
            T_c: final iteration of cosine annealing
    ''' 
    
    if t < T_w: 
        return (t / T_w) * a_max 
    elif t < T_c: 
        return a_min + (0.5 * (1 + math.cos(((t - T_w) / (T_c - T_w)) * math.pi)) * (a_max - a_min))
    else: 
        return a_min