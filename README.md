![uva-logo](src/uva-logo.jpg)

# Applied Machine Learning 2024

Lab assignments for the course Applied Machine Learning at the University of Amsterdam.

## week 1

week_1/Intro.ipynb

| function             | args                   |
| -------------------- | ---------------------- |
| w1_linear_forward    | ['x_input', 'P']       |
| w1_cal_pseudoinverse | ['x_input', 'y_input'] |
| w1_L2_regression     | ['x_input', 'y_input'] |

## week 2

week_2/ML.ipynb

| function                    | args                                       |
| --------------------------- | ------------------------------------------ |
| w2_linear_forward           | ['x_input', 'W', 'b']                      |
| w2_linear_grad_W            | ['x_input', 'grad_output', 'W', 'b']       |
| w2_linear_grad_b            | ['x_input', 'grad_output', 'W', 'b']       |
| w2_sigmoid_forward          | ['x_input']                                |
| w2_sigmoid_grad_input       | ['x_input', 'grad_output']                 |
| w2_nll_forward              | ['target_pred', 'target_true']             |
| w2_nll_grad_input           | ['target_pred', 'target_true']             |
| w2_dist_to_training_samples | ['x_input', 'training_set']                |
| w2_nearest_neighbors        | ['distances', 'training_labels']           |
| w2_tree_weighted_entropy    | ['Y_left', 'Y_right', 'classes']           |
| w2_tree_split_data_left     | ['X', 'Y', 'feature_index', 'split_value'] |
| w2_tree_split_data_right    | ['X', 'Y', 'feature_index', 'split_value'] |
| w2_tree_to_terminal         | ['Y']                                      |

## week 3

week_3/Neural_Nets.ipynb

| function           | args                        |
| ------------------ | --------------------------- |
| w3_dense_forward   | ['x_input', 'W', 'b']       |
| w3_relu_forward    | ['x_input']                 |
| w3_l2_regularizer  | ['weight_decay', 'weights'] |
| w3_conv_matrix     | ['matrix', 'kernel']        |
| w3_box_blur        | ['image', 'box_size']       |
| w3_maxpool_forward | ['x_input']                 |
| w3_flatten_forward | ['x_input']                 |

## list

w1_linear_forward
w1_cal_pseudoinverse
w1_L2_regression
w2_linear_forward
w2_linear_grad_W
w2_linear_grad_b
w2_sigmoid_forward
w2_sigmoid_grad_input
w2_nll_forward
w2_nll_grad_input
w2_dist_to_training_samples
w2_nearest_neighbors
w2_tree_weighted_entropy
w2_tree_split_data_left
w2_tree_split_data_right
w2_tree_to_terminal
w3_dense_forward
w3_relu_forward
w3_l2_regularizer
w3_conv_matrix
w3_box_blur
w3_maxpool_forward
w3_flatten_forward
