import pandas
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
def data_preprocessing(task_1a_dataframe):
    encoded_dataframe = task_1a_dataframe.drop(['JoiningYear','City', 'Age'], axis=1)
    
    
    le = LabelEncoder()
    encoded_dataframe['Education'] = le.fit_transform(encoded_dataframe['Education'])
    encoded_dataframe['Gender'] = le.fit_transform(encoded_dataframe['Gender'])
    encoded_dataframe['EverBenched'] = le.fit_transform(encoded_dataframe['EverBenched'])
    

    return encoded_dataframe

def identify_features_and_targets(encoded_dataframe):

    features = encoded_dataframe[['Education','PaymentTier','Gender','EverBenched','ExperienceInCurrentDomain']]

    target = encoded_dataframe['LeaveOrNot']

    features_and_targets = [features, target]

    return features_and_targets


def load_as_tensors(features_and_targets):

    features, target = features_and_targets

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    tensors_and_iterable_training_data = [X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, train_loader]

    return tensors_and_iterable_training_data

class Salary_Predictor(nn.Module):

    def __init__(self):

        super(Salary_Predictor, self).__init__()

        input_size = 5
        hidden_size1 = 198
        hidden_size2 = 120
        hidden_size3 = 89
        hidden_size4 = 98
        output_size = 1  
        
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3) 
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, hidden_size4) 
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_size4, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)        
        out = self.sigmoid(out)
        return out

def model_loss_function():
    
    loss_function = nn.BCELoss()
      
    return loss_function

def model_optimizer(model):
    
    learning_rate = 0.0001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    return optimizer

def model_number_of_epochs():
   
    number_of_epochs = 200

    return number_of_epochs

def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
    
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, train_loader = tensors_and_iterable_training_data

    for epoch in range(number_of_epochs):
        model.train() 
        total_loss = 0.0

        for batch_X, batch_y in train_loader:

            optimizer.zero_grad()

            outputs = model(batch_X)

            loss = loss_function(outputs, batch_y.view(-1, 1)) 

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{number_of_epochs}], Loss: {avg_loss:.4f}')

    return model

def validation_function(trained_model, tensors_and_iterable_training_data):
    
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, _ = tensors_and_iterable_training_data

    trained_model.eval()

    with torch.no_grad():
        y_pred = trained_model(X_test_tensor)

    y_pred_binary = (y_pred >= 0.5).float()
    
    for i in range(len(y_pred_binary)):
        print(f"Validation Sample {i + 1}: Predicted = {int(y_pred_binary[i][0])}, Actual = {int(y_test_tensor[i])}")

    correct = (y_pred_binary == y_test_tensor.view(-1, 1)).sum().item()
    total = len(y_test_tensor)
    model_accuracy = correct / total

    return model_accuracy

if __name__ == "__main__":

    task_1a_dataframe = pandas.read_csv('employee_dataset.csv')

    encoded_dataframe = data_preprocessing(task_1a_dataframe)

    features_and_targets = identify_features_and_targets(encoded_dataframe)

    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)

    model = Salary_Predictor()

    loss_function = model_loss_function()
    optimizer = model_optimizer(model)
    number_of_epochs = model_number_of_epochs()

    trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer)

    model_accuracy = validation_function(trained_model,tensors_and_iterable_training_data)
    print(f"Accuracy on the test set = {model_accuracy}")

    X_train_tensor = tensors_and_iterable_training_data[0]
    x = X_train_tensor[0]
    jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "model.pth")
