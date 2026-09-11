import matplotlib.pyplot as plt

# Datos reales de la nueva ejecución de la configuración principal
train_losses = [
    4.8651, 1.7593, 1.6300, 1.5781, 1.5434, 1.5401, 1.5362, 1.4956, 1.5132, 1.4948,
    1.4740, 1.4833, 1.4683, 1.4765, 1.4722, 1.3988, 1.4627, 1.4203, 1.4465, 1.4338,
    1.4563, 1.4347, 1.4217, 1.4054, 1.4083, 1.4183, 1.3812, 1.4310, 1.3966, 1.3823
]

val_losses = [
    4.8607, 1.6902, 1.5783, 1.5368, 1.5214, 1.5213, 1.4879, 1.5301, 1.5185, 1.4713,
    1.4696, 1.4412, 1.5065, 1.4840, 1.4371, 1.4375, 1.4605, 1.4381, 1.4239, 1.4277,
    1.4242, 1.4284, 1.4154, 1.4392, 1.4209, 1.4232, 1.4079, 1.4100, 1.4152, 1.4314
]

steps = list(range(0, 3000, 100))

plt.figure(figsize=(10, 6))
plt.plot(steps, train_losses, label='Training Loss', color='blue', linewidth=2, marker='o', markersize=4)
plt.plot(steps, val_losses, label='Validation Loss', color='orange', linewidth=2, marker='s', markersize=4)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss – Main Transformer Configuration', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('loss_curves_train_val.png', dpi=300)
plt.show()
print("✅ Gráfica guardada como loss_curves_train_val.png")