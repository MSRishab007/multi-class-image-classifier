import base64
import shutil
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import web_app1.app as app_module


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()

    def test_index_route_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Prediction and Retrieval", response.data)

    @patch.object(app_module, "load_model_bundle")
    @patch.object(app_module, "predict_image")
    def test_predict_route_renders_results(self, mock_predict_image, mock_load_model_bundle):
        mock_load_model_bundle.return_value = object()
        mock_predict_image.return_value = (
            "cup",
            {
                "color": "red",
                "material": "plastic",
                "condition": "new",
                "size": "small",
            },
        )

        response = self.client.post(
            "/predict",
            data={
                "model_name": "deit",
                "dataset_variant": "ours",
                "image": (BytesIO(b"fake image bytes"), "sample.jpg"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"cup", response.data)
        self.assertIn(b"red", response.data)

    @patch.object(app_module, "load_model_bundle")
    @patch.object(app_module, "retrieve_images")
    def test_retrieve_route_renders_results(self, mock_retrieve_images, mock_load_model_bundle):
        mock_load_model_bundle.return_value = object()
        token = base64.urlsafe_b64encode(b"sample.jpg").decode("ascii")
        mock_retrieve_images.return_value = [
            {
                "image_token": token,
                "class_name": "cup",
                "attributes": {
                    "color": "red",
                    "material": "plastic",
                    "condition": "new",
                    "size": "small",
                },
            }
        ]

        response = self.client.post(
            "/retrieve",
            data={
                "model_name": "deit",
                "dataset_variant": "ours",
                "query": "cup",
                "top_k": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"cup", response.data)
        self.assertIn(b"red", response.data)

    def test_data_route_serves_local_file(self):
        temp_dir = tempfile.mkdtemp()
        try:
            temp_path = Path(temp_dir) / "sample.jpg"
            temp_path.write_bytes(b"hello world")

            token = base64.urlsafe_b64encode(b"sample.jpg").decode("ascii")
            with patch.object(app_module, "DATA_DIR", temp_dir), patch.object(
                app_module, "DATA_DIR_REALPATH", str(Path(temp_dir).resolve())
            ):
                response = self.client.get(f"/data/{token}")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.data, b"hello world")
                response.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
