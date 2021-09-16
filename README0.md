App start:
python run.py

App deploy:
gcloud app deploy dev.yaml -v v1
gcloud app deploy prod.yaml -v v1

View logs:
gcloud app logs tail --service logger-processer -v v1