"""S3/MinIO storage for model checkpoints and weight submissions.

All large binary blobs (trained weights, checkpoints, datasets) are stored
in S3-compatible object storage rather than the database or memory.
"""

import io

import boto3
import structlog
from botocore.config import Config

from nexus.config import settings

logger = structlog.get_logger()

_client = None
_public_client = None


def _get_client():
    """Client for internal S3 operations (upload, download, delete)."""
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
    return _client


def _get_public_client():
    """Client for generating presigned URLs that external users can reach.

    If s3_public_url is set, presigned URLs use the public endpoint.
    Otherwise falls back to the internal endpoint.
    """
    global _public_client
    if _public_client is None:
        endpoint = settings.s3_public_url or settings.s3_endpoint_url
        _public_client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
    return _public_client


def ensure_bucket() -> None:
    """Create the bucket if it doesn't exist."""
    client = _get_client()
    try:
        client.head_bucket(Bucket=settings.s3_bucket)
    except client.exceptions.ClientError:
        client.create_bucket(Bucket=settings.s3_bucket)
        logger.info("s3_bucket_created", bucket=settings.s3_bucket)


def upload_bytes(key: str, data: bytes) -> str:
    """Upload raw bytes to S3 and return the key."""
    client = _get_client()
    client.put_object(Bucket=settings.s3_bucket, Key=key, Body=data)
    logger.debug("s3_upload", key=key, size=len(data))
    return key


def download_bytes(key: str) -> bytes:
    """Download raw bytes from S3."""
    client = _get_client()
    resp = client.get_object(Bucket=settings.s3_bucket, Key=key)
    data = resp["Body"].read()
    logger.debug("s3_download", key=key, size=len(data))
    return data


def delete_object(key: str) -> None:
    """Delete an object from S3."""
    client = _get_client()
    client.delete_object(Bucket=settings.s3_bucket, Key=key)


def generate_presigned_url(key: str, expires_in: int = 3600) -> str:
    """Generate a presigned download URL using the public S3 endpoint.

    If S3_PUBLIC_URL is not set, the URL will use the internal endpoint
    and only work from within the same network (e.g. Railway internal).
    """
    client = _get_public_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket, "Key": key},
        ExpiresIn=expires_in,
    )
