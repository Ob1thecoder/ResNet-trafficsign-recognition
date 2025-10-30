
"""
Created on Wed Aug 17 21:23:37 2022
"""
import torch
import numpy as np

def evaluate(model, testloader):
    """
    Parameters
    ----------
    model : ResNet model
        DESCRIPTION: 
            It should get the img as input and generate the prediction class
            as output like follow:   output = model(img)
            output must be a probability matrix with 43 values for classification 
    testloader : The defined dataloader class
        DESCRIPTION:
            The testloader is used to get the test img and its annotation for 
            accuracy generation.
    Return:  Output the test accuracy
    -------
    """
    device = next(model.parameters()).device  # Get device from model
    model.eval()
    
    # Move counters to the same device as the model
    total_count = torch.tensor([0.0]).to(device)
    correct_count = torch.tensor([0.0]).to(device)
    class_count = np.zeros(43)
    class_sum = np.array([60, 720, 750, 450, 660, 630, 150, 450, 450, 480, 660, 420, 690, 720, 270, 210, 150, 360, 390,  60,  90,  90, 120, 150, 90, 480, 180, 60, 150, 90, 150, 270, 60, 210, 120, 390, 120, 60, 690, 90, 90, 60, 90])
    evaluate_results = []
    
    with torch.no_grad():
        for i, data in enumerate(testloader):
            img, label, filename = data
            
            # Move data to device
            img = img.to(device)
            label = label.to(device)
            
            total_count += label.size(0)
            output = model(img)
            predict = torch.argmax(output, dim=1)
            correct_count += (predict == label).sum()
            
            for j in range(label.size(0)):
                if label[j] == predict[j]:
                    class_count[label[j].cpu().numpy()] += 1
                else:
                    evaluate_results.append([label[j].cpu().numpy(), filename[j]])
                
    testing_accuracy = correct_count / total_count
    print("Each class accuracy:\n", class_count / class_sum * 100)
    return testing_accuracy.item()