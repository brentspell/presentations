import rsa

p = 11
q = 13
e, d = rsa.keys(p, q)
print("primes:      {0}, {1}".format(p, q))
print("public key:  {0}".format(e))
print("private key: {0}".format(d))
print()

p = "encrypt me"
c = bytes(rsa.encipher(ord(p), e) for p in p)
r = ''.join(chr(rsa.decipher(c, e, d)) for c in c)
print("plaintext:  {0}".format(p))
print("enciphered: {0}".format(c))
print("deciphered: {0}".format(r))
print()

p = "verify me"
s = bytes(rsa.sign(ord(p), e, d) for p in p)
v = all(rsa.verify(s, e) == ord(p) for p, s in zip(p, s))
print("message:    {0}".format(p))
print("signature:  {0}".format(s))
print("verified:   {0}".format(v))
