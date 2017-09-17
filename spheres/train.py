import os
from unet3d.model import unet_model_3d
from unet3d.training import load_old_model, train_model
from unet3d.generator import ExampleSphereGenerator

input_shape=(1, 144,144,144)
model = unet_model_3d(input_shape=input_shape)

train_generator, validation_generator = ExampleSphereGenerator.get_training_and_validation(input_shape, cnt=5)

train_model(model=model, 
            model_file=os.path.abspath("./SphereCNN.h5"), 
            training_generator=train_generator,
            validation_generator=validation_generator, 
            steps_per_epoch=train_generator.num_steps,
            validation_steps=validation_generator.num_steps, 
            initial_learning_rate=0.00001,
            learning_rate_drop=0.5,
            learning_rate_epochs=10, 
            n_epochs=50)