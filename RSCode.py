import galois
import numpy as np

class RSCode:
    def __init__(self, m,t,l,m0):
        self.m = m #GF(2^m) field
        self.t = t #Error correction capability
        self.n = 2**m-1 #Code length
        self.k = self.n-2*t #Information length
        self.l = l #Shortened information length (-> shortened code length = l+n-k)
        self.m0 = m0 #m0 of the Reed-Solomon code, determines first root of generator
        
        self.g = self.makeGenerator(m,t,m0) # generator polynomial represented by a galois.Poly variable

    def encode(self,msg):
        # Systematically encodes information words using the Reed-Solomon code
        # Input:
        #  -msg: a 2D array of galois.GF elements, every row corresponds with a GF(2^m) information word of length self.l
        # Output:
        #  -code: a 2D array of galois.GF elements, every row contains a GF(2^m) codeword corresponding to systematic Reed-Solomon coding of the corresponding information word
        assert np.shape(msg)[1] == self.l, 'the number of columns must be equal to self.l'
        assert type(msg) is galois.GF(2**self.m) , 'each element of msg  must be a galois.GF element'

        GF = galois.GF(2**self.m)
        nrows = np.shape(msg)[0]
        npar = self.n - self.k
        code = GF.Zeros((nrows, self.l + npar))

        for i in range(nrows):
            # shift msg poly by x^(n-k) then take remainder with g(x)
            buf = GF.Zeros(self.l + npar)
            buf[:self.l] = msg[i]
            rem = galois.Poly(buf) % self.g
            # galois strips leading zeros so we pad back
            rc = rem.coeffs
            par = GF.Zeros(npar)
            par[npar - len(rc):] = rc
            code[i, :self.l] = msg[i]
            code[i, self.l:] = par

        assert np.shape(code)[1] == self.l+self.n-self.k , 'the number of columns must be equal to self.l+self.n-self.k'
        assert type(code) is galois.GF(2**self.m) , 'each element of code  must be a galois.GF element'
        return code

    def decode(self,code):
        # Decode Reed-Solomon codes
        # Input:
        #  -code: a 2D array of galois.GF elements, every row contains a GF(2^m) codeword of length self.l+self.n-self.k
        # Output:
        #  -decoded: a 2D array of galois.GF elements, every row contains a GF(2^m) information word corresponding to decoding of the corresponding Reed-Solomon codeword
        #  -nERR: 1D numpy array containing the number of corrected symbols for every codeword, -1 if error correction failed
        assert np.shape(code)[1] == self.l+self.n-self.k , 'the number of columns must be equal to self.l+self.n-self.k'
        assert type(code) is galois.GF(2**self.m) , 'each element of code  must be a galois.GF element'

        GF = galois.GF(2**self.m)
        N, clen = np.shape(code)
        tt = 2 * self.t
        a = GF.primitive_element

        decoded = GF.Zeros((N, self.l))
        nERR = np.zeros(N, dtype=int)

        # need x^2t for modding later (forney)
        xt_c = GF.Zeros(tt + 1)
        xt_c[0] = GF(1)
        xt = galois.Poly(xt_c)

        for row in range(N):
            w = code[row]
            rpoly = galois.Poly(w)

            # syndromes
            syn = GF.Zeros(tt)
            for j in range(tt):
                syn[j] = rpoly(a**(self.m0 + j))

            if np.all(syn == 0):
                decoded[row] = w[:self.l]
                nERR[row] = 0
                continue

            # PGZ algorithm - try from max errors down
            lam = None
            nerr = 0
            for v in range(self.t, 0, -1):
                # syndrome matrix
                mat = GF.Zeros((v, v))
                for i in range(v):
                    for j in range(v):
                        mat[i, j] = syn[i + j]
                rhs = GF.Zeros(v)
                for i in range(v):
                    rhs[i] = syn[v + i]
                try:
                    x = np.linalg.solve(mat, rhs)
                except np.linalg.LinAlgError:
                    continue
                lc = GF.Zeros(v + 1)
                lc[:v] = x
                lc[v] = GF(1)  # constant term
                lam = galois.Poly(lc)
                nerr = v
                break

            if lam is None:
                decoded[row] = w[:self.l]
                nERR[row] = -1
                continue

            # chien search
            errloc = []
            for i in range(clen):
                if lam(a**(-i)) == 0:
                    errloc.append(i)

            if len(errloc) != nerr:
                decoded[row] = w[:self.l]
                nERR[row] = -1
                continue

            # forney for error magnitudes
            spoly = galois.Poly(syn[::-1])
            omega = (spoly * lam) % xt
            lam_d = lam.derivative()

            fix = GF(np.array(w))
            bad = False
            for d in errloc:
                xi = a**(-d)
                dd = lam_d(xi)
                if dd == 0:
                    bad = True
                    break
                ev = (a**d)**(1 - self.m0) * omega(xi) / dd
                fix[clen - 1 - d] = fix[clen - 1 - d] + ev

            if bad:
                decoded[row] = w[:self.l]
                nERR[row] = -1
                continue

            # check if it actually worked
            cpoly = galois.Poly(fix)
            valid = True
            for j in range(tt):
                if cpoly(a**(self.m0 + j)) != 0:
                    valid = False
                    break
            if valid:
                decoded[row] = fix[:self.l]
                nERR[row] = nerr
            else:
                decoded[row] = w[:self.l]
                nERR[row] = -1

        assert np.shape(decoded)[1] == self.l, 'the number of columns must be equal to self.l'
        assert type(decoded) is galois.GF(2**self.m) , 'each element of decoded  must be a galois.GF element'
        assert type(nERR) is np.ndarray and len(np.shape(nERR))==1 , 'nERR must be a 1D numpy array'

        return (decoded,nERR)




    @staticmethod
    def makeGenerator(m, t, m0):
        # Generate the Reed-Solomon generator polynomial with error correcting capability t over GF(2^m)
        # Input:
        #  -m: order of the galois field is 2^m
        #  -t: error correction capability of the Reed-Solomon code
        #  -m0: determines the first root of the generator polynomial
        # Output:
        #  -generator: generator polynomial represented by a galois.Poly variable

        GF = galois.GF(2**m)
        a = GF.primitive_element
        # multiply (x - a^i) for all roots
        gen = galois.Poly([1], field=GF)
        for i in range(m0, m0 + 2*t):
            root = GF([1, int(a**i)])  # (x + a^i), same as (x - a^i) in char 2
            gen = gen * galois.Poly(root)
        generator = gen

        assert type(generator) == type(galois.Poly([0],field=galois.GF(2**m))), 'generator must be a galois.Poly object'
        return generator

    @staticmethod
    def test():
        # function that illustrates how the other code of this class can be tested
        m0 = 0 # CD standard uses m0=0 (roots alpha^0, alpha^1, alpha^2, alpha^3)
        m=8
        t=5
        l=10
        rs = RSCode(m,t,l,m0) # Construct the RSCode object
        p=2
        prim_poly=galois.primitive_poly(p,m)
        galois_field=galois.GF(p**m,prim_poly)


        msg = galois_field(np.random.randint(0,2**8-1,(5,10))) # Generate a random message of 5 information words

        code = rs.encode(msg) # Encode this message

        # Introduce errors
        code[1,[2, 17]] = code[1,[4, 17]]+galois_field(1)
        code[2,7] = 0;
        code[3,[3, 1, 18, 19, 5]] = np.random.randint(0,2**8-1,(1,5))
        code[4,[3, 1, 18, 19, 5, 12]] = np.random.randint(0,2**8-1,(1,6))


        [decoded,nERR] = rs.decode(code) # Decode


        print(nERR)
        assert((decoded[0:4,:] == msg[0:4,:]).all())
        pass