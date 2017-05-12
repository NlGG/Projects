def gam(datas):
    r = pyper.R(use_pandas='True')

    r.assign("data", datas)

    r("library(splines)")
    
    nd =len(datas)
    
    xx1 = np.array(datas['X'])
    xx2 = np.array(datas['Y'])

    lambda1a = 6.93e3
    lambda2a = 6.93e3
    
    sp1 = 1e3
    sp2 = 1e3
    
    p1 = np.linspace(min(datas['X']), max(datas['X']), 1000)
    nb1 = len(p1)
    hat1a = np.array([np.zeros(nb1) for i in range(nb1)])
    s0 = 0
    spar1 = s0 + 0.0601*log(sp1)
    for jj in range(nb1):
        y1 = np.zeros(nb1)
        y1[jj] = 1
        
        r.assign("p1", p1)
        r.assign("y1", y1)
        r.assign("spar1", spar1)
        
        r("ey1a <- smooth.spline(p1, y1, spar=spar1, all.knots=TRUE)$y")
        ey1a = r.get("ey1a")
        hat1a[:, jj] = ey1a
    
    r.assign("nb1", nb1)
    r("rr1 <- ns(p1, knots=p1[2:(nb1-1)],intercept=TRUE)")
    rr1 = r.get("rr1")
    pen1 = (rr1.T * np.linalg.inv(hat1a) * rr1 - rr1.T * rr1)/sp1
    r.assign("xx1", xx1)
    r("rr1a <- ns(c(0, xx1, 1), knots=p1[2:(nb1-1)],intercept=TRUE)")
    rr1a = r.get("rr1a")
    rr1a = rr1a[1:-1,]
    rr1a = np.mat(rr1a)
    pen1 = np.mat(pen1)
    hat1b = rr1a * np.linalg.inv(rr1a.T * rr1a + lambda1a*pen1)*rr1a.T
    
    p2 = np.linspace(min(datas['Y']), max(datas['Y']), 1000)
    nb2 = len(p2)
    hat2a = np.array([np.zeros(nb2) for i in range(nb2)])
    s0 = 0
    spar2 = s0 + 0.0601*log(sp2)
    for jj in range(nb2):
        y2 = np.zeros(nb2)
        y2[jj] = 1
        
        r.assign("p2", p2)
        r.assign("y2", y2)
        r.assign("spar2", spar2)
        
        r("ey2a <- smooth.spline(p2, y2, spar=spar2, all.knots=TRUE)$y")
        ey2a = r.get("ey2a")
        hat2a[:, jj] = ey2a
    
    r.assign("nb2", nb2)
    r("rr2 <- ns(p2, knots=p2[2:(nb2-1)],intercept=TRUE)")
    rr2 = r.get("rr2")
    pen2 = (rr1.T * np.linalg.inv(hat2a) * rr2 - rr2.T * rr2)/sp2
    r.assign("xx2", xx2)
    r("rr2a <- ns(c(0, xx2, 1), knots=p1[2:(nb2-1)],intercept=TRUE)")
    rr2a = r.get("rr2a")
    rr2a = rr2a[1:-1,]
    rr2a = np.mat(rr2a)
    pen2 = np.mat(pen2)
    hat2b = rr2a * np.linalg.inv(rr2a.T * rr2a + lambda2a*pen2)*rr2a.T
    
    yy = np.mat(datas['P']).T
    ey2 = yy
    for kk in range(5000):
        ey1 = hat1b * (yy - ey2)
        ey2 = hat2b * (yy - ey1)
    
    ey1av = np.mean(ey1)
    ey1 = ey1 - ey1av
    ey2 = ey2 + ey1av
    return ey1, ey2, ey1av