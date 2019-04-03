% binary-encode a message
message = "Hi"
plaintext = str2num(dec2bin(toascii(message))(:))'

% generate keys as inverse matrices
identity = eye(length(plaintext));
permutation = randperm(length(plaintext))

public = [identity(permutation, :)]
private = [identity(:, permutation)]

assert(public * private == identity)

% encrypt a message
ciphertext = plaintext * public

% decrypt the cipher text
recovered = ciphertext * private
plaintext
