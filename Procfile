build: ./rex
  command: gunicorn rex.wsgi:application
  expose:
    - 8000
  environment:
    - SECRET_KEY=please_change_me
    - SQL_ENGINE=django.db.backends.postgresql
    - SQL_DATABASE=postgres
    - SQL_USER=postgres
    - SQL_PASSWORD=postgres
    - SQL_HOST=db
    - SQL_PORT=5432
    - DATABASE=postgres
    - USE_S3=TRUE
    - AWS_ACCESS_KEY_ID=UPDATE_ME
    - AWS_SECRET_ACCESS_KEY=UPDATE_ME
    - AWS_STORAGE_BUCKET_NAME=UPDATE_ME
  depends_on:
    - db