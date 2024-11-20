# %% [markdown]
# # import libraries

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import itertools

# Sklearn
from sklearn.model_selection import train_test_split, \
                                GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# %% [markdown]
# # Read data

# %%
park_data = pd.read_csv("parkinsons.csv")

park_inp = park_data.drop("target", axis=1)
park_tar = park_data["target"]

# %% [markdown]
# # Basic data visualisation and scale analysis

# %% [markdown]
# ### Target visualisation for curiosity
#
# It's interesting to note that the scale varies from 0 to 260.
# Although as we will see in the target visualisation we only
# have scores below

# %%
plt.hist(park_tar)
#plt.xlim([0,260])

# %%
# Assuming park_data is already loaded

# Calculate min, max, and difference for each column
min_values = park_data.min()
max_values = park_data.max()
diff_values = max_values - min_values

# Create a DataFrame with min, max, and difference
summary_df = pd.DataFrame({
    'Min': min_values,
    'Max': max_values,
    'Difference': diff_values
})

# Display the summary
print("Min, Max, and Difference values for each column:")
print(summary_df)

# Create histograms for all columns
fig, axes = plt.subplots(5, 4, figsize=(20, 25))
fig.suptitle("Histograms of all features", fontsize=16)

for i, column in enumerate(park_data.columns):
    row = i // 4
    col = i % 4
    sns.histplot(park_data[column], ax=axes[row, col], kde=True)
    axes[row, col].set_title(column)
    axes[row, col].set_xlabel('')

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()

# %% [markdown]
# # 5) Model comparison
# - Linear regression Model (LR)
# - MLP Regressor with 2 hidden layers of 10 neurons each
#           + no activation functions (MLP_NA)
# - MLP Regressor with 2 hidden layers of 10 neurons each
#           + ReLU activation functions (MLP_ReLU)
#
#
#
# MLP random_state=0
# Boxplot of Mean Absolute Error (MAE) for each model

# %% [markdown]
# ### Data splitting

# %%
rand_st = list(range(1,11))

train_test_data = []
for rand in rand_st:
    inp_train_temp, inp_test_temp, tar_train_temp, tar_test_temp =\
    train_test_split(park_inp,park_tar,test_size=0.2,random_state=rand)
    train_test_data.append(
        [inp_train_temp, inp_test_temp, tar_train_temp, tar_test_temp])

# %% [markdown]
# Model Definition

# %%
# Linear Regression Model
linear_model = LinearRegression()

# MLP Regressor with 2 hidden layers, no activation
mlp_no_activation = MLPRegressor(hidden_layer_sizes=(10, 10),
                                 activation='identity',
                                 random_state=0)

# MLP Regressor with 2 hidden layers, ReLU activation
mlp_relu = MLPRegressor(hidden_layer_sizes=(10, 10),
                        activation='relu',
                        random_state=0)

# List of models
models = [linear_model, mlp_no_activation, mlp_relu]
model_names = ['Linear Regression', 'MLP (No Activation)', 'MLP (ReLU)']

# %%
def evaluate_model(model, X_train, y_train, X_test):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    return model.predict(X_test)

# Perform cross-validation and calculate MAE for each model
mae_scores = []

for model in models:
    mae_score_temp = []
    for run in range(10):
        y_pred = evaluate_model(model,
                                train_test_data[run][0],
                                train_test_data[run][2],
                                train_test_data[run][1])
        mae = mean_absolute_error(train_test_data[run][3], y_pred)
        mae_score_temp.append(mae)
        # Negate because sklearn returns negative MAE
    mae_scores.append(mae_score_temp)

# %% [markdown]
# Plot the results from the training.

# %%
plt.boxplot(mae_scores, labels=model_names)
plt.ylabel("MAE")

# %% [markdown]
# # 6)
# The results from the MLP_NA and the Linear Regression
# are very similar, this is because without an activation
# function the MLP produces only a Linear Regression.
# From We can think of the Results that come up from the MLP Like
#
#
# Z^{[1]} = W^{[1]}X^{[0]} + b^{[1]}
#
# X^{[1]} = Z^{[1]}
#
# Z^{[2]} = W^{[2]}X^{[1]} + b^{[2]}
#
# X^{[2]} = Z^{[2]} = W^{[2]}W^{[1]}X^{[0]} + W^{[2]}b^{[1]} + b^{[2]}
#
# Where the left half could also be represented as
# W and the right half a general constant

# %% [markdown]
# # 7.
# 20-80 train-test split random_state = 0
#
# Grid Search of hyperparameters from the model
# of MultiLayer Perceptron 2 hidden layers 10 neurons each
# - (i)   L2 penalty    [0.001,0.01,0.1]
# - (ii)  learning rate [0.001,0.01,0.1]
# - (iii) batch size    [32,64,128]

# %% [markdown]
# # 7)  Normal version

# %%
# Split the data
X_train_notscaled, X_test_notscaled, y_train_notscaled, y_test_notscaled =\
        train_test_split(park_inp,park_tar,test_size=0.2,random_state=0)



# Define the model and parameter grid
param_grid_notscaled = {
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128]
}
# Generate all combinations of hyperparameters
param_combinations = list(itertools.product(*param_grid_notscaled.values()))


# Initialize matrix to store results
mae_matrix_notscaled = np.zeros((3, 3, 3))

# Iterate through all combinations
for i, (alpha, learning_rate, batch_size) in enumerate(param_combinations):
    # Create and train the model
    model = MLPRegressor(hidden_layer_sizes=(10, 10),
                         random_state=0,alpha=alpha,
                         learning_rate_init=learning_rate, batch_size=batch_size)
    model.fit(X_train_notscaled, y_train_notscaled)

    # Make predictions on the test set
    y_pred = model.predict(X_test_notscaled)

    # Calculate MAE
    mae = mean_absolute_error(y_test_notscaled, y_pred)

    # Store MAE in the matrix
    mae_matrix_notscaled[
        param_grid_notscaled['alpha'].index(alpha),
        param_grid_notscaled['learning_rate_init'].index(learning_rate),
        param_grid_notscaled['batch_size'].index(batch_size)] = mae



# %%
# Find the best combination
best_idx = np.unravel_index(np.argmin(mae_matrix_notscaled),
                            mae_matrix_notscaled.shape)
best_params_notscaled = {
  'alpha': param_grid_notscaled['alpha'][best_idx[0]],
  'learning_rate_init': param_grid_notscaled['learning_rate_init'][best_idx[1]],
  'batch_size': param_grid_notscaled['batch_size'][best_idx[2]]
}
best_score_notscaled = mae_matrix_notscaled[best_idx]

# Plot the results
fig_notscaled, axes_notscaled = plt.subplots(1, 3, figsize=(20, 6))
batch_sizes_notscaled = [32, 64, 128]
# Find global min and max for consistent color scaling
value_min_notscaled = np.min(mae_matrix_notscaled)
value_max_notscaled = np.max(mae_matrix_notscaled)

for i, batch_size in enumerate(batch_sizes_notscaled):
    ax = axes_notscaled[i]
    im = ax.imshow(mae_matrix_notscaled[:, :, i], cmap='YlOrRd_r',
                   vmin=value_min_notscaled, vmax=value_max_notscaled)
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels([0.001, 0.01, 0.1], fontsize=14)
    ax.set_yticklabels([0.0001, 0.001, 0.01], fontsize=14)
    ax.set_xlabel('Learning Rate', fontsize=14)
    if i == 0: ax.set_ylabel('L2 Penalty (malpha)', fontsize=15)
    ax.set_title(f'Batch Size: {batch_size}', fontsize=14)

    for j in range(3):
        for k in range(3):
            text = ax.text(k, j, f'{mae_matrix_notscaled[j, k, i]:.3f}',
                           ha="center", va="center", color="black",
                           fontsize=17)

plt.tight_layout()
fig_notscaled.subplots_adjust(right=0.9)

# Add a colorbar to the right of the subplots
cbar_ax = fig_notscaled.add_axes([0.92, 0.15, 0.02, 0.7])
fig_notscaled.colorbar(im, cax=cbar_ax, label='MAE')

plt.show()

print("Best parameters:", best_params_notscaled)
print("Best test MAE:", best_score_notscaled)


