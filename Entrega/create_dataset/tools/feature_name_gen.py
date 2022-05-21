
def gen_name(base_name,n_features):
    names = []
    if type(n_features) is tuple:
        for r in range(n_features[0]):
            for c in range(n_features[1]):
                names.append("{name}_r{r}_c{c}".format(name=base_name, r=r, c=c))
    else:
        for i in range(n_features):
            names.append("{name}_{number}".format(name=base_name, number=i))
    return names
