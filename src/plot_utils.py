import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

def save_losses(output_dir, train_losses, eval_losses):
    with open(os.path.join(output_dir, 'losses.json'), 'w') as f:
        json.dump({'train_losses': train_losses, 'eval_losses': eval_losses}, f)
    print("Loss data saved successfully.")

def plot_and_save_losses(output_dir, train_losses, eval_losses):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(eval_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    print("Loss graph saved successfully.")
    
def plot_and_save_accuracies(output_dir, train_accuracies, eval_accuracies):
    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(eval_accuracies, label='Eval Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Train and Eval Accuracies')
    plt.savefig(os.path.join(output_dir, 'accuracies.png'))
    plt.close()