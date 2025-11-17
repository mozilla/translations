# Translations database

The database serves as a view for the data uploaded to Google Cloud Storage bucket by the training pipeline and for the corresponding Taskcluster tasks.

It includes metadata for training experiments, model artifacts, configs, evaluation results and training corpus.

It is used by the UI apps like [Model registry](https://mozilla.github.io/translations/model-registry).

The updater can recreate the database from scratch, but by default it uses it as a cache, so adding only recent runs is very fast.

The SQLite database is downloaded from the same GCS bucket and then uploaded back after the update.
This is mostly because we do not want to manage and maintain a proper persistent DB in the cloud. 
Also, we want it to be easily accessible from web apps directly.

The cron job to update the DB (and the dependent dashboards) runs daily.

## Running locally

To recreate the DB from scratch:

```bash
export PYTHONPATH=$(pwd) 
python db/updater.py --db-path data/db/db.sqlite --override
```
To upload to the production GCS bucket (requires write permissions):
```bash
gcloud auth application-default login
python db/updater.py --upload
```

To test the model registry UI:
```bash
python db/updater.py --db-path data/db/db.sqlite
cp data/db/db.sqlite site/data/db/
task serve-site
```

Then go to this link and the web app will load the local DB:
http://localhost:8080/model-registry/?db=/data/db/db.sqlite




