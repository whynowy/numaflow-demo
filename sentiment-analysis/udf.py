import sys
import uuid
import json
import logging

from pynumaflow.mapper import Messages, Message, Datum, Mapper
from transformers import pipeline


class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = pipeline("sentiment-analysis")

    def inference(self, keys: list[str], datum: Datum) -> Messages:
        strs = datum.value.decode("utf-8")
        messages = Messages()
        if len(strs) == 0:
            messages.append(Message.to_drop())
            return messages

        output = {'id': str(uuid.uuid4())}
        logging.info("%s - Received msg: %s", output['id'], strs)

        sentiment = self.analyzer(strs)
        output['text'] = strs
        output['sentiment'] = sentiment[0]['label']

        logging.info("%s - Sentiment analysis: %s", output['id'], output)
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

        logging.info("%s - Received msg: %s", output['id'], output)
        sentiment = output['sentiment']
        logging.info("%s - Sending msg to %s sink", output['id'], sentiment)

        output = json.dumps(output).encode("utf-8")
        messages.append(Message(keys=keys, value=output, tags=[sentiment]))
        return messages


if __name__ == "__main__":
    handler = sys.argv[1]
    logging.info("Starting handler: %s", handler)

    sa = SentimentAnalyzer()
    if handler == "inference":
        grpc_server = Mapper(handler=sa.inference)
        grpc_server.start()
    elif handler == "postprocess":
        grpc_server = Mapper(handler=sa.postprocess)
        grpc_server.start()
