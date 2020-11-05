import json
import mmh3
import flask
from flask import Flask, request
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-conversational")
model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-conversational", return_dict=True)

app = Flask(__name__)

for par in model.parameters():
    par.requires_grad = False

class RobertaClassifier(torch.nn.Module):
    def __init__(self):
        super(RobertaClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(393216, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = model(**x)
        x = x.last_hidden_state.view(x.last_hidden_state.size()[0], -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


classifier = RobertaClassifier()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0005)


def train(input, label):
    model.train()
    output = classifier(input)
    loss = criterion(output, label)
    optimizer.zero_grad()
    loss.backward()
    print(loss.item())
    optimizer.step()
    return output


@app.route("/", methods=['POST'])
def main():
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """

    response = {
        "version": request.json['version'],
        "session": request.json['session'],
        "response": {
            "end_session": False
        }
    }

    if request.json['session']['new']:
        response['response']['text'] = "Привет подписчикам YouTube канала andreiliphd! Добро пожаловать в навык графовый кот. \
                                Это приложение использует нейронную сеть в \
                                качестве демонстрации. Нейронная сеть не была \
                                натренирована. Используется PyTorch версии " + torch.__version__
        return json.dumps(
            response,
            ensure_ascii=False,
            indent=2
        )

    request_json = request.json
    input = tokenizer(request_json['request']['original_utterance'], return_tensors="pt", padding='max_length', max_length=512)
    output = train(input, torch.tensor(1).reshape(1))
    argmax = torch.argmax(output, dim=1)
    response['response']['text'] = "Категория классификации " + str(int(argmax))
    return json.dumps(
        response,
        ensure_ascii=False,
        indent=2
    )

@app.route("/health", methods=['GET'])
def health():
    return flask.Response(status=200)
