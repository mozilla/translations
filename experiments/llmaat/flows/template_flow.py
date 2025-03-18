# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os

from metaflow import (
    FlowSpec,
    IncludeFile,
    Parameter,
    card,
    current,
    step,
    environment,
    kubernetes,
)
from metaflow.cards import Markdown

GCS_PROJECT_NAME = "project-name-here"
GCS_BUCKET_NAME = "bucket-name-here"


class TemplateFlow(FlowSpec):
    """
    This flow is a template for you to use
    for orchestration of your model.
    """

    # You can import the contents of files from your file system to use in flows.
    # This is meant for small files—in this example, a bit of config.
    example_config = IncludeFile("example_config", default="./example_config.json")

    # This is an example of a parameter. You can toggle this when you call the flow
    # with python template_flow.py run --offline False
    offline_wandb = Parameter(
        "offline",
        help="Do not connect to W&B servers when training",
        type=bool,
        default=True,
    )

    # You can uncomment and adjust this decorator when it's time to scale your flow remotely.
    # @kubernetes(image="url-to-docker-image:tag", cpu=1)
    # Check https://docs.metaflow.org/api/step-decorators/kubernetes for details on @kubernetes decorator
    @card(type="default")
    @step
    def start(self):
        """
        Each flow has a 'start' step.

        You can use it for collecting/preprocessing data or other setup tasks.
        """

        self.next(self.train)

    @card
    @environment(
        vars={
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
            "WANDB_PROJECT": os.getenv("WANDB_PROJECT"),
        }
    )
    # You can uncomment and adjust this decorator to scale your flow remotely with a custom image.
    # Note: the image parameter must be a fully qualified registry path otherwise Metaflow will default to
    # the AWS public registry.
    # The image referenced HERE is the mozmlops demo image,
    # which has both the dependencies you need run this template flow:
    # scikit-learn (for the specific model called in this demo) and mozmlops (for all your ops tools).
    # Check https://docs.metaflow.org/api/step-decorators/kubernetes for details on @kubernetes decorator
    @kubernetes(image="registry.hub.docker.com/chelseatroy/mozmlops:latest", cpu=1)
    @step
    def train(self):
        """
        In this step you can train your model,
        save checkpoints and artifacts,
        and deliver data to Weights and Biases
        for experiment evaluation
        """
        import json
        import wandb
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from mozmlops.cloud_storage_api_client import CloudStorageAPIClient  # noqa: F401

        # This can help you fetch and upload artifacts to
        # GCS. Check out help(CloudStorageAPIClient) for more details.
        # It does require the account you're running the flow from to have
        # access to Google Cloud Storage.
        # storage_client = CloudStorageAPIClient(
        #     project_name=GCS_PROJECT_NAME, bucket_name=GCS_BUCKET_NAME
        # )

        config_as_dict = json.loads(self.example_config)
        print(f"The config file says: {config_as_dict.get('example_key')}")

        if not self.offline_wandb:
            tracking_run = wandb.init(project=os.getenv("WANDB_PROJECT"))
            wandb_url = tracking_run.get_url()
            current.card.append(Markdown("# Weights & Biases"))
            current.card.append(Markdown(f"Your training run is tracked [here]({wandb_url})."))

        print("All set. Running training.")
        # Model training goes here; for example, a LogisticRegression model on the iris dataset.
        # Of course, replace this example with YOUR model training code :).

        # Load the Iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the classifier
        clf = LogisticRegression(max_iter=300)

        # Train the classifier on the training data
        clf.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = clf.predict(X_test)

        prediction_path = os.path.join(  # noqa: F841
            current.flow_name, current.run_id, "y_predictions.txt"
        )
        observed_path = os.path.join(current.flow_name, current.run_id, "y_test.txt")  # noqa: F841

        # Example: How you'd store a checkpoint in the cloud
        predictions_for_storage = bytearray(y_pred)  # noqa: F841
        # storage_client.store(data=predictions_for_storage, storage_path=prediction_path)
        observed_values_for_storage = bytearray(y_test)  # noqa: F841
        # storage_client.store(
        #     data=observed_values_for_storage, storage_path=observed_path
        # )

        # Example: How you'd fetch a checkpoint from the cloud
        # storage_client.fetch(
        #     remote_path=prediction_path, local_path="y_predictions.txt"
        # )

        self.next(self.end)

    @step
    def end(self):
        """
        This is the mandatory 'end' step: it prints some helpful information
        to access the model and the used dataset.
        """
        print(
            f"""
            Flow complete.

            See artifacts at {GCS_BUCKET_NAME}.
            """
        )


if __name__ == "__main__":
    TemplateFlow()
