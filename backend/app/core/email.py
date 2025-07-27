import aiosmtplib
from email.message import EmailMessage

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "ecosyncinfo@gmail.com "
SMTP_PASSWORD = "vglg zlic vtmo emqo"

async def send_email(to_email: str, subject: str, body: str):
    message = EmailMessage()
    message["From"] = SMTP_USER
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(body)

    await aiosmtplib.send(
        message,
        hostname=SMTP_HOST,
        port=SMTP_PORT,
        start_tls=True,
        username=SMTP_USER,
        password=SMTP_PASSWORD,
    )
