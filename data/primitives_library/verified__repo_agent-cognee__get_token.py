import datetime
import os

import jwt

SECRET_KEY = os.getenv("FASTAPI_USERS_JWT_SECRET", "super_secret")


def create_jwt(user_id: str, tenant_id: str, roles: list[str]):
    claims = {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "roles": roles,
        "exp": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=1),
    }
    return jwt.encode(claims, SECRET_KEY, algorithm="HS256")


if __name__ == "__main__":
    token = create_jwt(
        "6763554c-91bd-432c-aba8-d42cd72ed659",
        "4523544d-82bd-432c-aca7-d42cd72ed651",
        ["admin"],
    )
    print(token)