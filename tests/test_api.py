import unittest

# setting path
import sys

sys.path.append("../dattabot")

from src.api import DattaBotAPI, DattaBotAPIException, DattaBotAPIResponse, Tensor
from src.agent_config import get_agent_config


class TestDattaBotAPI(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        # Setup agent config.
        self.config = get_agent_config()
        self.datta_bot_api = DattaBotAPI()

    def test_smoke_dattabot(self):
        self.assertEqual(1, 1)

    def test_tensor_encoding(self):
        # One query.
        one_query = ["Helloooo!"]
        resp: DattaBotAPIResponse = self.datta_bot_api.get_tensor_encoding(one_query)
        self.assertIsNotNone(resp)
        self.assertIsNotNone(resp.tensor_response)
        self.assertIsInstance(resp.tensor_response, Tensor)
        self.assertEqual(resp.tensor_response.size(dim=0), len(one_query))
        # Two queries.
        two_queries = ["Hello!", "We've met already."]
        resp: DattaBotAPIResponse = self.datta_bot_api.get_tensor_encoding(two_queries)
        self.assertIsNotNone(resp)
        self.assertIsNotNone(resp.tensor_response)
        self.assertIsInstance(resp.tensor_response, Tensor)
        self.assertEqual(resp.tensor_response.size(dim=0), len(two_queries))

    def test_respond_to_queries(self):
        three_queries = ["Hello!", "We've met already.", "Okay, nice to meet again."]
        resp: DattaBotAPIResponse = self.datta_bot_api.respond_to_queries(three_queries)
        self.assertIsNotNone(resp)

    # TODO(PiyushDatta): Get this test working.
    # def test_error_response_to_queries(self) -> str:
    #     queries = ["Hello world"]
    #     self.assertRaises(
    #         DattaBotAPIException,
    #         self.datta_bot_api._get_agent_action(queries=queries, action_type=-1),
    #     )


if __name__ == "__main__":
    unittest.main()