build: ./rex
  command: gunicorn rex.wsgi:application
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
    - AWS_ACCESS_KEY_ID=AKIAINO2E2WAHUYRTL4Q
    - AWS_SECRET_ACCESS_KEY=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
    - AWS_STORAGE_BUCKET_NAME=myrexbucket
