#!/usr/bin/env python3
"""
Authentication Middleware for Titans Finance API

This module provides authentication and authorization middleware
for the FastAPI application, including API key validation,
JWT token handling, and role-based access control.
"""

import os
import jwt
import time
import hashlib
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from functools import wraps

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
import redis
import logging

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

class AuthenticationError(Exception):
    """Custom authentication error"""
    pass

class AuthorizationError(Exception):
    """Custom authorization error"""
    pass

class APIKeyAuth:
    """API Key authentication handler"""

    def __init__(self, api_keys: List[str], redis_client: Optional[redis.Redis] = None):
        self.api_keys = set(api_keys)
        self.redis_client = redis_client
        self.failed_attempts = {}
        self.max_attempts = 5
        self.lockout_duration = 300  # 5 minutes

    def verify_api_key(self, api_key: str, request: Request) -> bool:
        """Verify API key and check rate limiting"""
        client_ip = self._get_client_ip(request)

        # Check if IP is locked out
        if self._is_locked_out(client_ip):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed authentication attempts. Try again later."
            )

        # Verify API key
        if api_key not in self.api_keys:
            self._record_failed_attempt(client_ip)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )

        # Reset failed attempts on successful authentication
        self._reset_failed_attempts(client_ip)

        # Log successful authentication
        self._log_authentication(api_key, client_ip, success=True)

        return True

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _is_locked_out(self, client_ip: str) -> bool:
        """Check if IP is locked out due to failed attempts"""
        if self.redis_client:
            lockout_key = f"auth_lockout:{client_ip}"
            return self.redis_client.exists(lockout_key)
        else:
            # In-memory fallback
            if client_ip in self.failed_attempts:
                attempts, last_attempt = self.failed_attempts[client_ip]
                if attempts >= self.max_attempts:
                    if time.time() - last_attempt < self.lockout_duration:
                        return True
                    else:
                        # Lockout period expired
                        del self.failed_attempts[client_ip]
        return False

    def _record_failed_attempt(self, client_ip: str):
        """Record a failed authentication attempt"""
        if self.redis_client:
            # Use Redis for distributed rate limiting
            attempt_key = f"auth_attempts:{client_ip}"
            lockout_key = f"auth_lockout:{client_ip}"

            attempts = self.redis_client.incr(attempt_key)
            self.redis_client.expire(attempt_key, self.lockout_duration)

            if attempts >= self.max_attempts:
                self.redis_client.setex(lockout_key, self.lockout_duration, "locked")
                logger.warning(f"IP {client_ip} locked out due to {attempts} failed attempts")
        else:
            # In-memory fallback
            current_time = time.time()
            if client_ip in self.failed_attempts:
                attempts, _ = self.failed_attempts[client_ip]
                self.failed_attempts[client_ip] = (attempts + 1, current_time)
            else:
                self.failed_attempts[client_ip] = (1, current_time)

    def _reset_failed_attempts(self, client_ip: str):
        """Reset failed attempts for successful authentication"""
        if self.redis_client:
            attempt_key = f"auth_attempts:{client_ip}"
            self.redis_client.delete(attempt_key)
        else:
            if client_ip in self.failed_attempts:
                del self.failed_attempts[client_ip]

    def _log_authentication(self, api_key: str, client_ip: str, success: bool):
        """Log authentication attempt"""
        # Hash API key for security
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:8]
        status_msg = "SUCCESS" if success else "FAILED"
        logger.info(f"API Auth {status_msg}: key={api_key_hash} ip={client_ip}")

class JWTAuth:
    """JWT token authentication handler"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client

    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

        return encoded_jwt

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})

        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

        # Store refresh token in Redis if available
        if self.redis_client:
            user_id = data.get("sub")
            if user_id:
                self.redis_client.setex(
                    f"refresh_token:{user_id}",
                    int(timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS).total_seconds()),
                    encoded_jwt
                )

        return encoded_jwt

    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

            # Check token type
            if payload.get("type") != token_type:
                raise AuthenticationError(f"Invalid token type. Expected {token_type}")

            # Check if token is blacklisted
            if self._is_token_blacklisted(token):
                raise AuthenticationError("Token has been revoked")

            return payload

        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.JWTError:
            raise AuthenticationError("Invalid token")

    def refresh_access_token(self, refresh_token: str) -> str:
        """Refresh access token using refresh token"""
        try:
            payload = self.verify_token(refresh_token, token_type="refresh")
            user_id = payload.get("sub")

            if not user_id:
                raise AuthenticationError("Invalid refresh token")

            # Check if refresh token exists in Redis
            if self.redis_client:
                stored_token = self.redis_client.get(f"refresh_token:{user_id}")
                if not stored_token or stored_token.decode() != refresh_token:
                    raise AuthenticationError("Refresh token not found or invalid")

            # Create new access token
            new_token_data = {
                "sub": user_id,
                "username": payload.get("username"),
                "permissions": payload.get("permissions", [])
            }

            return self.create_access_token(new_token_data)

        except AuthenticationError:
            raise
        except Exception as e:
            raise AuthenticationError(f"Token refresh failed: {str(e)}")

    def revoke_token(self, token: str):
        """Add token to blacklist"""
        if self.redis_client:
            # Extract expiration time from token
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
                exp = payload.get("exp")
                if exp:
                    ttl = exp - int(time.time())
                    if ttl > 0:
                        self.redis_client.setex(f"blacklist:{token}", ttl, "revoked")
            except jwt.JWTError:
                pass

    def _is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted"""
        if self.redis_client:
            return self.redis_client.exists(f"blacklist:{token}")
        return False

class RoleBasedAuth:
    """Role-based access control"""

    # Define permissions
    PERMISSIONS = {
        "read_transactions": "Read transaction data",
        "write_transactions": "Create/update transactions",
        "delete_transactions": "Delete transactions",
        "read_predictions": "Read ML predictions",
        "create_predictions": "Create ML predictions",
        "manage_models": "Manage ML models",
        "admin_access": "Administrator access"
    }

    # Define roles with their permissions
    ROLES = {
        "viewer": ["read_transactions", "read_predictions"],
        "analyst": ["read_transactions", "read_predictions", "create_predictions"],
        "editor": ["read_transactions", "write_transactions", "read_predictions", "create_predictions"],
        "admin": list(PERMISSIONS.keys())
    }

    @classmethod
    def check_permission(cls, user_permissions: List[str], required_permission: str) -> bool:
        """Check if user has required permission"""
        return required_permission in user_permissions or "admin_access" in user_permissions

    @classmethod
    def get_user_permissions(cls, user_roles: List[str]) -> List[str]:
        """Get all permissions for user roles"""
        permissions = set()
        for role in user_roles:
            if role in cls.ROLES:
                permissions.update(cls.ROLES[role])
        return list(permissions)

# Security dependency classes
security = HTTPBearer()

def verify_api_key(api_keys: List[str], redis_client: Optional[redis.Redis] = None):
    """Dependency for API key verification"""
    auth_handler = APIKeyAuth(api_keys, redis_client)

    def _verify(credentials: HTTPAuthorizationCredentials, request: Request):
        return auth_handler.verify_api_key(credentials.credentials, request)

    return _verify

def verify_jwt_token(redis_client: Optional[redis.Redis] = None):
    """Dependency for JWT token verification"""
    jwt_handler = JWTAuth(redis_client)

    def _verify(credentials: HTTPAuthorizationCredentials):
        payload = jwt_handler.verify_token(credentials.credentials)
        return payload

    return _verify

def require_permission(permission: str):
    """Decorator for permission-based access control"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user from kwargs (assumes JWT verification dependency)
            user_data = kwargs.get('current_user')
            if not user_data:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Authentication required"
                )

            user_permissions = user_data.get('permissions', [])
            if not RoleBasedAuth.check_permission(user_permissions, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{permission}' required"
                )

            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Utility functions
def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def generate_api_key() -> str:
    """Generate a secure API key"""
    import secrets
    import base64

    # Generate 32 random bytes and encode as base64
    random_bytes = secrets.token_bytes(32)
    api_key = base64.urlsafe_b64encode(random_bytes).decode('utf-8').rstrip('=')

    return f"tf_{api_key}"  # Prefix for Titans Finance

class AuthConfig:
    """Authentication configuration"""

    def __init__(self):
        self.api_keys = os.getenv("API_KEYS", "dev-api-key").split(",")
        self.jwt_secret = os.getenv("JWT_SECRET_KEY", SECRET_KEY)
        self.jwt_algorithm = os.getenv("JWT_ALGORITHM", ALGORITHM)
        self.access_token_expire = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", ACCESS_TOKEN_EXPIRE_MINUTES))
        self.refresh_token_expire = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", REFRESH_TOKEN_EXPIRE_DAYS))
        self.enable_rate_limiting = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"
        self.max_login_attempts = int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))
        self.lockout_duration = int(os.getenv("LOCKOUT_DURATION", "300"))

    def validate(self) -> bool:
        """Validate configuration"""
        if not self.api_keys or self.api_keys == ["dev-api-key"]:
            logger.warning("Using default API key. Change in production!")

        if self.jwt_secret == SECRET_KEY:
            logger.warning("Using default JWT secret. Change in production!")

        return True

# Simple auth dependency for our API routes
def verify_auth(credentials: HTTPAuthorizationCredentials = security):
    """Simple API key verification for ML Engineering API"""
    # Default API keys for development
    valid_api_keys = [
        "dev-api-key-change-in-production",
        "tf_development_key_123",
        "ml_engineering_key_456"
    ]
    
    if credentials.credentials not in valid_api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return credentials.credentials

# Example usage and testing
if __name__ == "__main__":
    # Test API key generation
    print("Generated API key:", generate_api_key())

    # Test password hashing
    password = "test_password"
    hashed = hash_password(password)
    print("Password verification:", verify_password(password, hashed))

    # Test JWT token creation
    jwt_auth = JWTAuth()
    token_data = {"sub": "user123", "username": "testuser", "permissions": ["read_transactions"]}
    access_token = jwt_auth.create_access_token(token_data)
    print("Created access token:", access_token[:50] + "...")

    # Test token verification
    try:
        payload = jwt_auth.verify_token(access_token)
        print("Token verification successful:", payload.get("username"))
    except AuthenticationError as e:
        print("Token verification failed:", e)
