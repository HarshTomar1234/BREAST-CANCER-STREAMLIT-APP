import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# pickle: This is the Python module used for serializing and deserializing Python objects.
# It allows you to convert Python objects into a byte stream that can be written to a file or transmitted over a network,
# and later reconstructed to its original form.

"""
*** Serializing: This is the process of converting a Python object into a byte stream, typically for the purpose of storing it to a file or sending it over a network.
Serialization allows you to preserve the state of an object so that it can be reconstructed later.
In the context of machine learning, serializing a trained model means converting it into a format that can be saved to disk and later loaded into memory to make predictions.


*** Deserializing: This is the opposite process of serialization. It involves reconstructing a Python object from a byte stream. 
Deserialization allows you to restore the original state of an object from its serialized form. 
In the context of machine learning, deserializing a model means loading a previously saved model from disk into memory so that it can be used for making predictions.

"""
# Serialization can be used for saving and restoring the state of an application. For example, in a web application, you might serialize user sessions to maintain state across multiple requests.


def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test model
    y_pred = model.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    return model, scaler


def get_clean_data():
    data = pd.read_csv("data/data.csv")

    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data


def main():
    data = get_clean_data()

    model, scaler = create_model(data)

    with open('model/model.pkl', 'wb') as f:
        # pickle.dump() is used to serialize a Python object and save it to a file.
        pickle.dump(model, f)

    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    main()
