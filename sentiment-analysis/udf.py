import sys

from pynumaflow.mapper import Messages, Message, Datum, Mapper
from transformers import pipeline
import json


class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = pipeline("sentiment-analysis")

    def inference(self, keys: list[str], datum: Datum) -> Messages:
        strs = datum.value.decode("utf-8")
        messages = Messages()
        if len(strs) == 0:
            messages.append(Message.to_drop())
            return messages

        output = {}
        sentiment = self.analyzer(strs)
        output['text'] = strs
        output['sentiment'] = sentiment[0]['label']
        output = json.dumps(output).encode("utf-8")
        messages.append(Message(output, keys=keys))
        return messages

    @staticmethod
    def postprocess(keys: list[str], datum: Datum) -> Messages:
        output = datum.value.decode("utf-8")
        messages = Messages()
        if len(output) == 0:
            messages.append(Message.to_drop())
            return messages

        sentiment = output['sentiment']
        output = json.dumps(output).encode("utf-8")

        messages.append(Message(keys=keys, value=output, tags=[sentiment]))
        return messages


if __name__ == "__main__":
    handler = sys.argv[1]
    sa = SentimentAnalyzer()
    if handler == "inference":
        grpc_server = Mapper(handler=sa.inference)
        grpc_server.start()
    elif handler == "postprocess":
        grpc_server = Mapper(handler=sa.postprocess)
        grpc_server.start()
