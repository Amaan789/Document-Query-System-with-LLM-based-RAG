import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
load_dotenv()

S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
BM25_ENCODER_FILE = os.environ.get("BM25_ENCODER_FILE")

s3_client = boto3.client(
    "s3",
    region_name="us-east-1"
)


def upload_bm25_to_s3(local_path: str):
    """
    Upload BM25 encoder file from local storage to S3
    """
    try:
        s3_client.upload_file(local_path, S3_BUCKET, BM25_ENCODER_FILE)
        print("BM25 file uploaded successfully to S3")
    except ClientError as e:
        print("Error uploading BM25 file:", str(e))
        raise


def download_bm25_from_s3(local_path: str):
    """
    Download BM25 encoder file from S3 to local storage (/tmp in Lambda)
    """
    try:
        s3_client.download_file(S3_BUCKET, BM25_ENCODER_FILE, local_path)
        print("BM25 file downloaded successfully from S3")
    except ClientError as e:
        print("Error downloading BM25 file:", str(e))
        raise