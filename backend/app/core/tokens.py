from itsdangerous import URLSafeTimedSerializer

SECRET_KEY = "food_secret"
SECURITY_PASSWORD_SALT = "food_secret_salt"

def generate_verification_token(email: str) -> str:
    serializer = URLSafeTimedSerializer(SECRET_KEY)
    return serializer.dumps(email, salt=SECURITY_PASSWORD_SALT)

def confirm_verification_token(token: str, expiration=3600) -> str | None:
    serializer = URLSafeTimedSerializer(SECRET_KEY)
    try:
        email = serializer.loads(token, salt=SECURITY_PASSWORD_SALT, max_age=expiration)
    except Exception:
        return None
    return email

def generate_password_reset_token(email: str) -> str:
    serializer = URLSafeTimedSerializer(SECRET_KEY)
    return serializer.dumps(email, salt="password-reset-salt")

def confirm_password_reset_token(token: str, expiration=3600) -> str | None:
    serializer = URLSafeTimedSerializer(SECRET_KEY)
    try:
        email = serializer.loads(token, salt="password-reset-salt", max_age=expiration)
    except Exception:
        return None
    return email

