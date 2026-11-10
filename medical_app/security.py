import hashlib
import hmac
import os
import secrets
import time


def password_hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()
    return f"{salt}:{digest}"


def verify_password(password, stored):
    return hmac.compare_digest(password_hash(password, stored.split(':')[0]), stored)


def new_login(store, user_id):
    token = secrets.token_urlsafe(32)
    store.execute("INSERT INTO logins VALUES(?,?,?)", (token_hash(token), user_id, time.time() + 86400))
    return token


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()
