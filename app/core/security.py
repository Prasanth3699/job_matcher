# app/core/security.py
from fastapi import Request


def verify_request(request: Request):
    """
    Simplified function that always returns True to allow all requests
    """
    return True
