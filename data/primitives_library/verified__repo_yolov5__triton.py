from __future__ import annotations

from urllib.parse import urlparse

import torch


class TritonRemoteModel:
    """A wrapper over a model served by the Triton Inference Server."""

    def __init__(self, url: str):
        """Initialize a Triton client from a server URL."""
        parsed_url = urlparse(url)

        if parsed_url.scheme == "grpc":
            from tritonclient.grpc import InferInput, InferenceServerClient

            self.client = InferenceServerClient(parsed_url.netloc)
            repository = self.client.get_model_repository_index()
            self.model_name = repository.models[0].name
            self.metadata = self.client.get_model_metadata(self.model_name, as_json=True)
        else:
            from tritonclient.http import InferInput, InferenceServerClient

            self.client = InferenceServerClient(parsed_url.netloc)
            repository = self.client.get_model_repository_index()
            self.model_name = repository[0]["name"]
            self.metadata = self.client.get_model_metadata(self.model_name)

        def create_input_placeholders() -> list[InferInput]:
            return [
                InferInput(
                    input_metadata["name"],
                    [int(dimension) for dimension in input_metadata["shape"]],
                    input_metadata["datatype"],
                )
                for input_metadata in self.metadata["inputs"]
            ]

        self._create_input_placeholders_fn = create_input_placeholders

    @property
    def runtime(self):
        """Return the model runtime."""
        return self.metadata.get("backend", self.metadata.get("platform"))

    def __call__(self, *args, **kwargs) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run inference using positional or named torch tensor inputs."""
        inputs = self._create_inputs(*args, **kwargs)
        response = self.client.infer(model_name=self.model_name, inputs=inputs)

        outputs = [
            torch.as_tensor(response.as_numpy(output_metadata["name"]))
            for output_metadata in self.metadata["outputs"]
        ]
        return outputs[0] if len(outputs) == 1 else outputs

    def _create_inputs(self, *args, **kwargs):
        """Create Triton input objects from positional or keyword tensor arguments."""
        positional_count = len(args)
        keyword_count = len(kwargs)

        if positional_count == 0 and keyword_count == 0:
            raise RuntimeError("No inputs provided.")
        if positional_count and keyword_count:
            raise RuntimeError("Cannot specify args and kwargs at the same time")

        placeholders = self._create_input_placeholders_fn()

        if positional_count:
            if positional_count != len(placeholders):
                raise RuntimeError(f"Expected {len(placeholders)} inputs, got {positional_count}.")
            for placeholder, value in zip(placeholders, args):
                placeholder.set_data_from_numpy(value.cpu().numpy())
        else:
            for placeholder in placeholders:
                placeholder.set_data_from_numpy(kwargs[placeholder.name].cpu().numpy())

        return placeholders