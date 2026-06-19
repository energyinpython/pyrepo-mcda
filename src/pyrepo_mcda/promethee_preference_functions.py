import numpy as np

# Preference function type 1 (Usual criterion) requires no parameters
# alternatives are indifferent only if they are equal to each other
# otherwise there is a strong preference for one of them
def preference_usual_function(d, p, q):
    if d <= 0:
        return 0
    else:
        return 1


# Preference function type 2 (U-shape criterion) requires indifference threshold (q)
def preference_ushape_function(d, p, q):
    if d <= q:
        return 0
    else:
        return 1


# Preference function type 3 (V-shape criterion) requires threshold of absolute preference (p)
def preference_vshape_function(d, p, q):
    if d <= 0:
        return 0
    elif 0 <= d <= p:
        return d / p
    elif d > p:
        return 1


# preference function type 4 (Level criterion) requires both preference and indifference thresholds (p and q)
def preference_level_function(d, p, q):
    if d <= q:
        return 0
    elif q < d <= p:
        return 0.5
    elif d > p:
        return 1


# Preference function type 5 (V-shape with indifference criterion also known as linear)
# requires both preference and indifference thresholds (p and q)
def preference_linear_function(d, p, q):
    if d <= q:
        return 0
    elif q < d <= p:
        return (d - q) / (p - q)
    elif d > p:
        return 1


# preference function type 6 (Gaussian criterion)
# requires to fix parameter s which is an intermediate value between q and p
def preference_gaussian_function(d, p, q):
    if d <= 0:
        return 0
    elif d > 0:
        s = (p + q) / 2
        return 1 - np.exp(-((d ** 2) / (2 * s ** 2)))
