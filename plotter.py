import matplotlib.pyplot as plt

def plot(H, str_label, save_directory):
	loss_values = H.history['loss']
	accuracy = H.history['accuracy']
	val_loss_values = H.history['val_loss']
	val_accuracy = H.history['val_accuracy']
	epochs = range(1, len(loss_values)+1)
	fig_H = plt.figure()
	plt.yticks(range(0, 2, 0.2))
	plt.plot(epochs, loss_values, label='Training Loss', figure=fig_H)
	plt.plot(epochs, accuracy, label='Training Accuracy', figure=fig_H)
	plt.plot(epochs, val_loss_values, label='Validation Loss', figure=fig_H)
	plt.plot(epochs, val_accuracy, label='Validation Accuracy', figure=fig_H)
	plt.title(str_label)
	plt.xlabel('Epochs')
	plt.legend()
	plt.savefig(save_directory + "\\" + str_label + ".png")