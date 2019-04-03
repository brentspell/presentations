# standard rsa public exponent
EXPONENT = 65537


# a little math
def gcd(a, b):
    return b if a % b == 0 else gcd(b, a % b)


def lcm(a, b):
    return a * b / gcd(a, b)


# key generation
def keys(prime_1, prime_2):
    # public key
    public = prime_1 * prime_2

    # euler's totient
    ϕ = lcm(prime_1 - 1, prime_2 - 1)

    # private key
    private = 1
    while private * EXPONENT % ϕ != 1:
        private += 1

    return public, private


# encryption
def encipher(plaintext, public):
    return plaintext ** EXPONENT % public


def decipher(ciphertext, public, private):
    return ciphertext ** private % public


# digital signatures
def sign(message, public, private):
    return message ** private % public


def verify(signature, public):
    return signature ** EXPONENT % public
