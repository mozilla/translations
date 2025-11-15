# Translations database

The database serves as a view for the data uploaded to Google Cloud Storage bucket by the training pipeline and for the corresponding Taskcluster tasks.

It includes training experiments, model artifacts, configs, evaluation results and training corpus.

It is used by the UI apps like Model registry.

The updater can recreate the database from scratch, but by default it uses it as a cache, so adding only recent runs is very fast.

The SQLite database is downloaded from the same GCS bucket and then uploaded back after the update.
As it should be updated on every export task, even if a race condition occurs, the next update will fix it.
This is mostly because we do not want to manage and maintain a proper persistent DB in the cloud.