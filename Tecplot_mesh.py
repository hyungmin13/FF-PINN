import numpy as np
def tecplot_Mesh(filename, X, Y, Z, x, y, z, vars, fw):
    #filename: path + *.dat  / X, Y, Z : data shape / x, y, z : (-1) numpy array  /  
    #vars : list or tuple object containing (-1) size numpy array and id number  /  fw : object length - ex) fw9 for type double
    def pad(s, width):
        s2 = s
        while len(s2) < width:
            s2 = ' ' + s2
        if s2[0] != ' ':
            s2 = ' ' + s2
        if len(s2) > width:
            s2 = s2[:width]
        return s2
    def varline(vars, id, fw):
        s = ""
        for i in range(len(vars)):
            s = s + pad(str(vars[i][1][id]),fw)
        s = s + '\n'
        return s

    f = open(filename, "wt")

    f.write('Variables="x [mm]", "y [mm]", "z [mm]"')
    for i in range(len(vars)):
        f.write(',"%s"' % vars[i][0])
    f.write('\n\n')
    f.write('Zone I=' + pad(str(X),6) + ',J=' + pad(str(Y),6) + ',K=' + pad(str(Z),6))
    f.write(', F=POINT\n')
    id = 0
    for i in range(vars[0][1].shape[0]):
        f.write(pad(str(np.float32(x[i])),fw) + pad(str(np.float32(y[i])),fw) + pad(str(np.float32(z[i])),fw))
        f.write(varline(vars, id, fw))
        id = id + 1

    f.close()
