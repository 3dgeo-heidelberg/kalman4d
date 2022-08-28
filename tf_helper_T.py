import numpy as np

def opk2R(om, ph, ka):
    """
    Euler rotation angles to rotation matrix according to Kraus, p.489
    :param om: Omega
    :param ph: Phi
    :param ka: Kappa
    :return: Rotation matrix
    """
    cos = np.cos
    sin = np.sin

    R = np.array([[cos(ph)*cos(ka)                        , -cos(ph)*sin(ka)                        ,  sin(ph)        ],
                  [cos(om)*sin(ka)+sin(om)*sin(ph)*cos(ka),  cos(om)*cos(ka)-sin(om)*sin(ph)*sin(ka), -sin(om)*cos(ph)],
                  [sin(om)*sin(ka)-cos(om)*sin(ph)*cos(ka),  sin(om)*cos(ka)+cos(om)*sin(ph)*sin(ka),  cos(om)*cos(ph)]]).T

    return R

def opkM2R(om, ph, ka, m):
    """
    Euler rotation angles to rotation matrix according to Kraus, p.489
    :param om: Omega
    :param ph: Phi
    :param ka: Kappa
    :param m: Scale (-1, as a factor, e.g. -0.004)
    :return: Rotation+scale matrix
    """
    R = opk2R(om, ph, ka) * (1 + m)
    return R

def dRdOPK_opk2R(om, ph, ka):

    cos = np.cos
    sin = np.sin
    dRdOm = np.array([[0                                       , 0                        ,  0        ],
                      [-sin(om)*sin(ka)+cos(om)*sin(ph)*cos(ka),  -sin(om)*cos(ka)-cos(om)*sin(ph)*sin(ka), -cos(om)*cos(ph)],
                      [cos(om)*sin(ka)+sin(om)*sin(ph)*cos(ka),  cos(om)*cos(ka)-sin(om)*sin(ph)*sin(ka),  -sin(om)*cos(ph)]]).T

    dRdPh = np.array([[-sin(ph)*cos(ka)                        , sin(ph)*sin(ka)                        ,  cos(ph)        ],
                      [sin(om)*cos(ph)*cos(ka)                 , sin(om)*cos(ph)*sin(ka),         sin(om)*sin(ph)],
                      [cos(om)*cos(ph)*cos(ka),  cos(om)*cos(ph)*sin(ka),  -cos(om)*sin(ph)]]).T

    dRdKa = np.array([[-cos(ph)*sin(ka)                        , cos(ph)*cos(ka)                        ,  0        ],
                      [cos(om)*cos(ka)-sin(om)*sin(ph)*sin(ka),  -cos(om)*sin(ka)-sin(om)*sin(ph)*cos(ka), 0],
                      [sin(om)*cos(ka)+cos(om)*sin(ph)*sin(ka),  -sin(om)*sin(ka)+cos(om)*sin(ph)*cos(ka),  0]]).T

    return dRdOm, dRdPh, dRdKa

def dRdOPKM_opkm2R(om, ph, ka, m):
    dRdOm, dRdPh, dRdKa = dRdOPK_opk2R(om, ph, ka)
    dRdM = opk2R(om, ph, ka)

    return dRdOm * (1+m), dRdPh * (1+m), dRdKa * (1+m), dRdM

def CxxOPK_XYZ2Cxx14(om, ph, ka, Cxx):
    dRdOm, dRdPh, dRdKa = dRdOPK_opk2R(om, ph, ka)

    F = np.block([[dRdOm.reshape(9,1), dRdPh.reshape(9,1), dRdKa.reshape(9,1), np.zeros((9,3))],
                  [np.zeros((3,1)),np.zeros((3,1)),np.zeros((3,1)), np.eye(3)]])
    Cxx_new = np.linalg.multi_dot([F, Cxx, F.T])
    return Cxx_new

def CxxOPKM_XYZ2Cxx14(om, ph, ka, m, Cxx):
    dRdOm, dRdPh, dRdKa, dRdM = dRdOPKM_opkm2R(om, ph, ka, m)

    F = np.block([[dRdOm.reshape(9,1), dRdPh.reshape(9,1), dRdKa.reshape(9,1), dRdM.reshape(9,1), np.zeros((9,3))],
                  [np.zeros((3,1)),np.zeros((3,1)),np.zeros((3,1)),np.zeros((3,1)), np.eye(3)]])
    Cxx_new = np.linalg.multi_dot([F, Cxx, F.T])
    return Cxx_new

def validate_MC(Cxx, tf, nSample=100000):
    from numpy.random import multivariate_normal
    inps = multivariate_normal(tf, Cxx, size=nSample)
    outps = np.zeros((12,nSample))
    for i in range(nSample):
        outps[:9, i] = opk2R(*inps[i, :-3]).reshape((9,))
        outps[9:, i] = inps[i, 3:]
    MC = np.cov(outps)
    EP = CxxOPK_XYZ2Cxx14(tf[0], tf[1], tf[2], Cxx)
    print(np.max(MC-EP), np.min(MC-EP))
    print(np.isclose(MC, EP))
    print(MC-EP)
    #tft = np.concatenate((opk2R(*tf[:3]).reshape(9), np.array(tf[3:])))
    diagErr = np.sqrt(np.diag(MC))-np.sqrt(np.diag(EP))
    tft = np.sqrt(np.diag(EP))
    pdiagErr = diagErr / tft * 100.
    print(pdiagErr)
    pass



if __name__ == '__main__':
    from scipy.io import loadmat
    data = loadmat(r'Cxx_example.mat')
    Cxx = data['Cxx']
    tf = data['xhat']
    Cxx_new = CxxOPK_XYZ2Cxx14(tf[0,0], tf[1,0], tf[2,0], Cxx)
    #print(Cxx_new, Cxx_new.shape)
    #print(np.all(np.isclose(Cxx_new - Cxx_new.T, np.zeros((12,12)))))
    #Cxx[5,5] = 0.01
    validate_MC(Cxx, [tf[0,0], tf[1,0], tf[2,0], tf[3,0], tf[4,0], tf[5,0]])

