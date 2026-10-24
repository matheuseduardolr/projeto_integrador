FROM python:3.12-slim

RUN pip install --no-cache-dir \
    "mlflow==3.16.1" \
    "sqlalchemy>=2.0,<2.1" \
    "psycopg2-binary>=2.9" \
    "boto3>=1.34"

EXPOSE 5010

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5010"]
