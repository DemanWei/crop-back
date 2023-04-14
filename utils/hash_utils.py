import hashlib

def encrypt(string):
    return hashlib.sha1(string.encode('utf-8')).hexdigest()